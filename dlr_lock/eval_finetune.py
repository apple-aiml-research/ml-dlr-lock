#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fine-tuning resistance: how fast can an attacker adapt the locked model?

Fine-tunes the pretrained baseline and the locked model under identical
conditions and records the evaluation loss against three cost axes:

* **tokens** -- sample efficiency, independent of hardware;
* **FLOPs**  -- ``6 N`` per token, or ``8 N`` with gradient checkpointing;
* **wall clock** -- what the attacker actually pays.

The gap on the token axis is the architectural-mismatch effect; the extra gap on
the wall-clock axis is the backward-pass overhead on top of it. Both matter, and
they should be read together.

Fine-tuning learning rate is the attacker's main lever, so sweep it: the paper
uses {3e-6, 1e-5, 3e-5, 1e-4} and compares each arm against the baseline at the
same rate. A fair comparison also reports each arm at *its own* best rate, since
an attacker would tune.

Example
-------
    # baseline
    python -m dlr_lock.eval_finetune \\
        --model Qwen/Qwen3-0.6B --dataset wikitext --lr 1e-5

    # locked
    python -m dlr_lock.eval_finetune \\
        --model Qwen/Qwen3-0.6B --dataset wikitext --lr 1e-5 \\
        --rank 32 --locked-dir runs/qwen3-0.6b/phase2
"""

import argparse
import json
import math
import os
import time

import torch

from .data import BatchIterator, build_finetune_data
from .loading import (autocast, load_causal_lm, load_tokenizer,
                      peak_memory_gb, reset_peak_memory)
from .model import build_locked_model, load_dlrnet_weights, set_checkpointing


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--locked-dir", default="",
                   help="Directory of DLR-Net weights. Empty means fine-tune "
                        "the unmodified pretrained model (the baseline arm).")
    p.add_argument("--rank", type=int, default=32,
                   help="Must match the rank the weights were trained with.")
    p.add_argument("--multiplier", type=float, default=1.0)

    p.add_argument("--dataset", default="wikitext",
                   choices=["wikitext", "c4", "alpaca"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-dataset-tokens", type=int, default=0,
                   help="Cap the training corpus. 0 streams effectively "
                        "unlimited data; a finite cap models a data-limited "
                        "attacker and makes the quality ceiling visible.")

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr-schedule", default="cosine",
                   choices=["cosine", "constant"])

    p.add_argument("--grad-checkpointing", action="store_true", default=True,
                   help="The attacker's best memory mitigation; on by default "
                        "so the comparison is against a competent attacker.")
    p.add_argument("--no-grad-checkpointing", action="store_false",
                   dest="grad_checkpointing")
    p.add_argument("--optim-8bit", action="store_true")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="",
                   help="e.g. 'auto' to shard a large model across GPUs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", default=None,
                   help="Pin the Hugging Face Hub revision (branch, tag, or "
                        "commit hash) for the model and dataset. Unset means "
                        "the default branch, whose contents can change; pin it "
                        "for reproducible runs.")

    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--early-eval-steps", default="1,2,3,4,5,6,7,8,9,10,"
                                                 "12,14,16,18,20,30,50,75",
                   help="Extra eval points early in training, where the "
                        "baseline and locked curves separate fastest.")
    p.add_argument("--output", default="",
                   help="Write the run history to this JSON path.")

    return p.parse_args(argv)


def finetune_flops_per_step(n_params, batch_size, seq_len,
                            grad_checkpointing=False):
    """``6 N`` per token, or ``8 N`` with checkpointing (forward recompute)."""
    return (8 if grad_checkpointing else 6) * n_params * batch_size * seq_len


def fmt_flops(f):
    for unit, scale in (("E", 1e18), ("P", 1e15), ("T", 1e12), ("G", 1e9)):
        if f >= scale:
            return f"{f / scale:.2f} {unit}FLOP"
    return f"{f:.0f} FLOP"


@torch.no_grad()
def evaluate(model, eval_batches, device):
    model.eval()
    total = 0.0
    for ids, labels in eval_batches:
        with autocast(device):
            total += model(input_ids=ids.to(device),
                           labels=labels.to(device)).loss.item()
    model.train()
    return total / max(len(eval_batches), 1)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    locked = bool(args.locked_dir)
    arm = "locked" if locked else "baseline"
    device_map = args.device_map or None

    print(f"=== Fine-tuning {arm} ===")
    print(f"  model={args.model} dataset={args.dataset} lr={args.lr:.0e} "
          f"seed={args.seed}")

    if locked:
        model = build_locked_model(
            args.model, args.rank, args.multiplier,
            device=args.device, device_map=device_map,
            torch_dtype=torch.bfloat16, revision=args.revision)
        load_dlrnet_weights(model, args.locked_dir)
        print(f"  Loaded DLR-Net weights from {args.locked_dir}")
    else:
        model = load_causal_lm(args.model, device=args.device,
                               device_map=device_map,
                               revision=args.revision)

    tokenizer = load_tokenizer(args.model, revision=args.revision)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if locked:
            set_checkpointing(model, True)
    elif locked:
        # Store every DLR-Net intermediate: the unmitigated Theta(L*d)
        # activation footprint.
        set_checkpointing(model, False)

    model.requires_grad_(True)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = finetune_flops_per_step(
        n_params, args.batch_size, args.seq_len, args.grad_checkpointing)
    print(f"  {n_params / 1e6:.1f}M params, "
          f"{fmt_flops(flops_per_step)}/step")

    print(f"  Building {args.dataset} data ...")
    train_examples = build_finetune_data(
        tokenizer, args.dataset, "train", args.seq_len, args.seed,
        args.max_dataset_tokens, revision=args.revision)
    val_examples = build_finetune_data(
        tokenizer, args.dataset, "validation", args.seq_len, args.seed,
        revision=args.revision)
    train_iter = BatchIterator(train_examples, args.batch_size,
                               args.device, args.seed + 1)
    eval_batches = [
        (torch.stack([e[0] for e in val_examples[i:i + args.batch_size]]),
         torch.stack([e[1] for e in val_examples[i:i + args.batch_size]]))
        for i in range(0, min(len(val_examples),
                              args.eval_batches * args.batch_size),
                       args.batch_size)
    ]
    print(f"  {len(train_examples)} train / {len(eval_batches)} eval batches")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optim_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay)

    warmup = max(100, int(args.steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return max(1e-7 / args.lr, step / warmup)
        if args.lr_schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    early = {int(s) for s in args.early_eval_steps.split(",") if s.strip()}
    history = {
        "arm": arm, "model": args.model, "dataset": args.dataset,
        "lr": args.lr, "seed": args.seed, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "grad_checkpointing": args.grad_checkpointing,
        "n_params": n_params, "flops_per_step": flops_per_step,
        "step": [], "tokens": [], "cum_flops": [], "wall_clock": [],
        "eval_loss": [], "peak_mem_gb": [],
    }

    def record(step, elapsed):
        loss = evaluate(model, eval_batches, args.device)
        peak = peak_memory_gb(args.device)
        history["step"].append(step)
        history["tokens"].append(step * args.batch_size * args.seq_len)
        history["cum_flops"].append(step * flops_per_step)
        history["wall_clock"].append(elapsed)
        history["eval_loss"].append(loss)
        history["peak_mem_gb"].append(peak)
        print(f"  [{arm}] step {step:5d}  eval_loss={loss:.4f}  "
              f"t={elapsed:.1f}s  peak={peak:.1f}GB")
        return loss

    reset_peak_memory(args.device)

    t0 = time.time()
    record(0, 0.0)
    t0 = time.time()  # exclude the step-0 eval from the wall clock

    for step in range(1, args.steps + 1):
        ids, labels = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device):
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0:
            print(f"  [{arm}] step {step:5d}  train_loss={loss.item():.4f}  "
                  f"{fmt_flops(step * flops_per_step)}")

        if step in early or step % args.eval_interval == 0:
            record(step, time.time() - t0)

    if history["step"][-1] != args.steps:
        record(args.steps, time.time() - t0)

    best = min(history["eval_loss"])
    print(f"\n[{arm}] best eval loss {best:.4f} at step "
          f"{history['step'][history['eval_loss'].index(best)]}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Wrote {args.output}")

    return history


if __name__ == "__main__":
    main()
