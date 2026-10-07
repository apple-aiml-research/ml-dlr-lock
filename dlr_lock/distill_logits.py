#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Phase 2: end-to-end logit distillation of the locked model.

Phase 1 fits each MLP sub-block independently, so its per-layer errors are small
but they compound across the depth of the transformer. This phase corrects the
accumulated drift by matching the full model's output distribution to the
pretrained teacher's, training only the DLR-Net parameters -- attention,
embeddings, and the output head stay frozen.

The loss is a top-k KL divergence: the sum is restricted to the k tokens the
teacher ranks highest, which cuts the logit traffic by three orders of magnitude
against a 150k vocabulary while preserving the part of the distribution that
carries the signal.

This phase does back-propagate through the whole locked model, so the defender
pays the same activation-memory cost the attacker will. It is paid once, and
Phase 1's initialization makes it converge quickly -- skipping Phase 1 is
Pareto-dominated at every compute budget.

Example
-------
    python -m dlr_lock.distill_logits \\
        --model Qwen/Qwen3-0.6B --rank 32 \\
        --phase1-dir runs/qwen3-0.6b/phase1 \\
        --output-dir runs/qwen3-0.6b/phase2 \\
        --steps 100000 --lr 1e-3 --topk 1000
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import build_text_stream
from .loading import (autocast, empty_cache, load_causal_lm,
                      load_tokenizer)
from .model import (compute_depth, dlrnet_config, replace_layer_mlp,
                    save_dlrnet_weights)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--phase1-dir", default="",
                   help="Directory of Phase-1 layer{ii}.pt files. If empty, "
                        "the DLR-Nets start from random init (the ablation "
                        "that skips Phase 1).")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--dataset", default="wikitext", choices=["wikitext", "c4"])
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)

    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=1000,
                   help="Restrict the KL sum to the teacher's top-k tokens. "
                        "0 uses the full vocabulary.")
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--teacher-device", default="cuda:0")
    p.add_argument("--student-device", default="cuda:0",
                   help="Set to a second GPU for large models; the teacher "
                        "runs forward-only and needs far less memory.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--optim-8bit", action="store_true",
                   help="8-bit AdamW via bitsandbytes, to fit the optimizer "
                        "state of a large model.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", default=None,
                   help="Pin the Hugging Face Hub revision (branch, tag, or "
                        "commit hash) for the model and dataset. Unset means "
                        "the default branch, whose contents can change; pin it "
                        "for reproducible runs.")

    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=2500)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=5000)

    return p.parse_args(argv)


@torch.no_grad()
def evaluate_ppl(model, stream, n_steps, device):
    """Perplexity as ``exp`` of the mean token cross-entropy."""
    was_training = model.training
    model.eval()
    total = 0.0
    with autocast(device):
        for _ in range(n_steps):
            ids = next(stream).to(device)
            total += model(input_ids=ids, labels=ids).loss.item()
    if was_training:
        model.train()
    return math.exp(total / max(n_steps, 1))


def topk_kl_loss(student_logits, teacher_logits, topk, temperature):
    """Top-k KL divergence between teacher and student next-token distributions.

    The teacher's top-k logits are renormalized into a distribution over just
    those k tokens, and the student is scored on the same index set. Dropping
    the tail is what makes this affordable: only k logits cross the device
    boundary instead of the whole vocabulary.
    """
    if temperature != 1.0:
        teacher_logits = teacher_logits / temperature
        student_logits = student_logits / temperature

    if topk > 0:
        k = min(topk, teacher_logits.shape[-1])
        topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
        teacher_lp = F.log_softmax(topk_vals.float(), dim=-1)
        teacher_lp = teacher_lp.to(student_logits.device)
        topk_idx = topk_idx.to(student_logits.device)
        student_lp = F.log_softmax(
            student_logits.gather(-1, topk_idx).float(), dim=-1)
    else:
        teacher_lp = F.log_softmax(
            teacher_logits.float(), dim=-1).to(student_logits.device)
        student_lp = F.log_softmax(student_logits.float(), dim=-1)

    loss = F.kl_div(student_lp, teacher_lp,
                    reduction="batchmean", log_target=True)
    if temperature != 1.0:
        loss = loss * temperature ** 2
    return loss


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    os.makedirs(args.output_dir, exist_ok=True)
    multi_gpu = args.teacher_device != args.student_device

    print(f"Loading student {args.model} on {args.student_device} ...")
    model = load_causal_lm(args.model, device=args.student_device,
                           revision=args.revision)
    model.requires_grad_(False)

    print(f"Loading teacher on {args.teacher_device} ...")
    teacher = load_causal_lm(args.model, device=args.teacher_device,
                             revision=args.revision)
    teacher.requires_grad_(False)
    teacher.eval()

    tokenizer = load_tokenizer(args.model, revision=args.revision)

    d = model.config.hidden_size
    d_ff = model.config.intermediate_size
    n_layers = len(model.model.layers)
    depth = compute_depth(d, d_ff, args.rank, args.multiplier)

    cfg = dlrnet_config(d, d_ff, args.rank, args.multiplier)
    print(f"  d={d} layers={n_layers} depth={depth}")
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if args.phase1_dir:
        print(f"\nInserting Phase-1 DLR-Nets from {args.phase1_dir} ...")
    else:
        print("\nInserting randomly initialized DLR-Nets "
              "(no Phase 1) ...")
    for li in range(n_layers):
        sd = None
        if args.phase1_dir:
            path = os.path.join(args.phase1_dir, f"layer{li:02d}.pt")
            sd = torch.load(path, map_location="cpu", weights_only=True)
        replace_layer_mlp(model, li, d, args.rank, depth, state_dict=sd,
                          alpha_init=0.0)
    print(f"  Replaced {n_layers} MLPs")

    # Gradient checkpointing at the transformer-layer level, on top of the
    # DLR-Nets' own internal checkpointing.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # Train the DLR-Nets only, in fp32 for optimizer stability.
    train_params = []
    for layer in model.model.layers:
        for p in layer.mlp.parameters():
            p.data = p.data.float()
            p.requires_grad_(True)
            train_params.append(p)
    print(f"  Trainable: {sum(p.numel() for p in train_params):,} params")

    if args.optim_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            train_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            train_params, lr=args.lr, weight_decay=args.weight_decay)

    warmup = max(100, int(args.steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return max(1e-7 / args.lr, step / warmup)
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.compile:
        print("  Compiling MLPs ...")
        for layer in model.model.layers:
            layer.mlp = torch.compile(layer.mlp)
        for layer in teacher.model.layers:
            layer.mlp = torch.compile(layer.mlp)

    print(f"\nBuilding {args.dataset} stream ...")
    train_stream = build_text_stream(
        tokenizer, args.dataset, "train", args.seq_len, args.batch_size,
        args.seed + 1, revision=args.revision)
    eval_stream = build_text_stream(
        tokenizer, args.dataset, "validation", args.seq_len, 4,
        args.seed + 2, revision=args.revision)

    baseline = evaluate_ppl(model, eval_stream, args.eval_steps,
                            args.student_device)
    print(f"  Baseline PPL (Phase-1 weights inserted): {baseline:.2f}")

    print(f"\nTraining {args.steps} steps ...")
    t0 = time.time()
    running = 0.0

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        ids = next(train_stream)
        teacher_ids = ids.to(args.teacher_device)
        student_ids = ids.to(args.student_device) if multi_gpu else teacher_ids

        with torch.no_grad(), autocast(args.teacher_device):
            teacher_logits = teacher(input_ids=teacher_ids).logits

        with autocast(args.student_device):
            student_logits = model(input_ids=student_ids).logits

        loss = topk_kl_loss(student_logits, teacher_logits,
                            args.topk, args.temperature)
        del teacher_logits, student_logits
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        running += loss.item()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            rate = step / elapsed
            print(f"  step {step:6d}/{args.steps} | "
                  f"KL={running / args.log_interval:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"{rate:.1f} step/s "
                  f"ETA {(args.steps - step) / rate / 60:.0f}min")
            running = 0.0

        if step % args.eval_interval == 0:
            ppl = evaluate_ppl(model, eval_stream, args.eval_steps,
                               args.student_device)
            print(f"  step {step:6d} | ppl={ppl:.2f}")
            empty_cache(args.student_device)

        if args.save_interval and step % args.save_interval == 0:
            save_dlrnet_weights(model, args.output_dir, n_layers)

    final = evaluate_ppl(model, eval_stream, args.eval_steps,
                         args.student_device)
    save_dlrnet_weights(model, args.output_dir, n_layers)
    print(f"\nFinal PPL: {final:.2f} (from {baseline:.2f})")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
