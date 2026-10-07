#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Phase 1: module-wise regression of each MLP sub-block into a DLR-Net.

For every target transformer layer ``l`` we run the frozen pretrained model
forward, capture the sub-block input ``z`` (the residual stream entering the
MLP branch) and its output ``f_l(z) = z + FFN_l(RMSNorm(z))``, and train a
DLR-Net ``g_l`` to match it under a relative mean-squared error

    L_l = E || g_l(z) - f_l(z) ||^2 / E || f_l(z) ||^2.

The normalization matters because output norms vary by an order of magnitude
across depth, so an unnormalized loss would silently prioritize late layers.

The layers are independent: each student sees only its own activations, so this
phase never back-propagates through the full model and its cost is linear in the
number of layers. Teacher activations are produced once per forward and fanned
out to every student being trained in that invocation.

Example
-------
    python -m dlr_lock.distill_modules \\
        --model Qwen/Qwen3-0.6B --rank 32 \\
        --output-dir runs/qwen3-0.6b/phase1 \\
        --steps 500000 --lr 1e-3
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import build_text_stream
from .loading import load_causal_lm, load_tokenizer
from .model import DLRNet, compute_depth, dlrnet_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="HuggingFace model id or local path.")
    p.add_argument("--rank", type=int, default=32,
                   help="Low-rank bottleneck r. Choose so that the resulting "
                        "depth lands in [140, 150].")
    p.add_argument("--multiplier", type=float, default=1.0,
                   help="Parameter budget relative to the SwiGLU MLP.")
    p.add_argument("--layers", default="all",
                   help="Comma-separated layer indices, a 'start:end' range, "
                        "or 'all'.")
    p.add_argument("--output-dir", required=True,
                   help="Where to write layer{ii}.pt and config.json.")

    p.add_argument("--dataset", default="wikitext",
                   choices=["wikitext", "c4"],
                   help="Distillation corpus. Match it to the model's own "
                        "pretraining data when you can; quality preservation "
                        "improves markedly.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Sequences per teacher forward.")
    p.add_argument("--student-batch-tokens", type=int, default=4096,
                   help="Tokens per student optimizer step. Each teacher "
                        "forward yields batch_size*seq_len tokens, split into "
                        "chunks of this size.")

    p.add_argument("--steps", type=int, default=500000,
                   help="Student optimizer steps.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--teacher-device", default="cuda:0")
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the students (recommended; the DLR-Net "
                        "is a long chain of small ops).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", default=None,
                   help="Pin the Hugging Face Hub revision (branch, tag, or "
                        "commit hash) for the model and dataset. Unset means "
                        "the default branch, whose contents can change; pin it "
                        "for reproducible runs.")

    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=5000)
    p.add_argument("--eval-tokens", type=int, default=10000)
    p.add_argument("--save-interval", type=int, default=50000)

    return p.parse_args(argv)


def parse_layers(spec, n_layers):
    if spec == "all":
        return list(range(n_layers))
    if ":" in spec:
        start, end = spec.split(":")
        return list(range(int(start), int(end)))
    return [int(x) for x in spec.split(",") if x.strip()]


class SubBlockCapture:
    """Forward hooks that record ``(z, z + FFN(RMSNorm(z)))`` per layer.

    Hooking the norm's input rather than the MLP's gives us ``z`` before
    normalization, which is what the DLR-Net consumes: it applies its own
    RMSNorm at every one of its layers.
    """

    def __init__(self, model, layer_indices, dtype=torch.bfloat16):
        self.layer_indices = list(layer_indices)
        self.dtype = dtype
        self.x = {li: None for li in self.layer_indices}
        self.y = {li: None for li in self.layer_indices}
        self._h = {li: None for li in self.layer_indices}
        self._handles = []

        for li in self.layer_indices:
            layer = model.model.layers[li]
            self._handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._make_pre(li), with_kwargs=True))
            self._handles.append(
                layer.mlp.register_forward_hook(self._make_post(li)))

    def _make_pre(self, li):
        def hook(mod, args, kwargs=None):
            h = args[0] if args else kwargs["hidden_states"]
            self._h[li] = h
            self.x[li] = h.detach().to(self.dtype).reshape(-1, h.shape[-1])
        return hook

    def _make_post(self, li):
        def hook(mod, args, output):
            out = output[0] if isinstance(output, tuple) else output
            y = self._h[li] + out
            self.y[li] = y.detach().to(self.dtype).reshape(-1, y.shape[-1])
            self._h[li] = None
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def collect_eval_activations(teacher, layer_indices, stream, n_tokens,
                             device):
    """Gather a fixed held-out activation set for validation."""
    xs = {li: [] for li in layer_indices}
    ys = {li: [] for li in layer_indices}
    got = 0
    with SubBlockCapture(teacher, layer_indices) as cap:
        while got < n_tokens:
            ids = next(stream).to(device)
            teacher(input_ids=ids, attention_mask=torch.ones_like(ids))
            for li in layer_indices:
                xs[li].append(cap.x[li].float().cpu())
                ys[li].append(cap.y[li].float().cpu())
            got += ids.numel()

    out = {}
    for li in layer_indices:
        x = torch.cat(xs[li])[:n_tokens]
        y = torch.cat(ys[li])[:n_tokens]
        # E||f(x)||^2, the denominator of the relative MSE.
        out[li] = (x, y, y.pow(2).mean().item())
    return out


def make_scheduler(optimizer, lr, total_steps, warmup_ratio):
    warmup = max(1000, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return max(1e-7 / lr, step / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.005 + 0.995 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading teacher {args.model} on {args.teacher_device} ...")
    teacher = load_causal_lm(args.model, device=args.teacher_device,
                             revision=args.revision)
    teacher.requires_grad_(False)
    teacher.eval()

    tokenizer = load_tokenizer(args.model, revision=args.revision)

    d = teacher.config.hidden_size
    d_ff = teacher.config.intermediate_size
    n_layers = len(teacher.model.layers)
    depth = compute_depth(d, d_ff, args.rank, args.multiplier)
    layer_indices = parse_layers(args.layers, n_layers)

    cfg = dlrnet_config(d, d_ff, args.rank, args.multiplier)
    print(f"  d={d} d_ff={d_ff} layers={n_layers}")
    print(f"  DLRNet: r={args.rank} depth={depth} "
          f"({cfg['params']:,} params, "
          f"{cfg['params'] / cfg['swiglu_params']:.2f}x SwiGLU)")
    print(f"  Training layers: {layer_indices}")
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nBuilding {args.dataset} stream ...")
    train_stream = build_text_stream(
        tokenizer, args.dataset, split="train", seq_len=args.seq_len,
        batch_size=args.batch_size, seed=args.seed + 1,
        revision=args.revision)
    eval_stream = build_text_stream(
        tokenizer, args.dataset, split="validation", seq_len=args.seq_len,
        batch_size=args.batch_size, seed=args.seed + 2,
        revision=args.revision)

    print("Collecting held-out activations ...")
    eval_data = collect_eval_activations(
        teacher, layer_indices, eval_stream, args.eval_tokens,
        args.teacher_device)
    for li in layer_indices:
        print(f"  layer {li:2d}: E||f(z)||^2 = {eval_data[li][2]:.6f}")

    students, optimizers, schedulers = {}, {}, {}
    for li in layer_indices:
        s = DLRNet(d, args.rank, depth, alpha_init=0.0).to(
            args.student_device, dtype=torch.float32)
        if args.compile:
            s = torch.compile(s, mode="reduce-overhead")
        s.train()
        students[li] = s
        optimizers[li] = torch.optim.AdamW(
            s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        schedulers[li] = make_scheduler(
            optimizers[li], args.lr, args.steps, args.warmup_ratio)

    def evaluate(li):
        s = students[li]
        x, y, norm_sq = eval_data[li]
        s.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(x), args.student_batch_tokens):
                xb = x[i:i + args.student_batch_tokens].to(args.student_device)
                yb = y[i:i + args.student_batch_tokens].to(args.student_device)
                total += F.mse_loss(s(xb), yb).item() * len(xb)
                n += len(xb)
        s.train()
        return (total / max(n, 1)) / norm_sq

    def save(li, step=None):
        sd = students[li].state_dict()
        sd = {k.removeprefix("_orig_mod."): v.cpu() for k, v in sd.items()}
        name = f"layer{li:02d}.pt" if step is None \
            else f"layer{li:02d}_step{step:07d}.pt"
        torch.save(sd, os.path.join(args.output_dir, name))

    print(f"\nTraining {args.steps} steps ...")
    t0 = time.time()
    running = {li: 0.0 for li in layer_indices}
    step = 0

    with SubBlockCapture(teacher, layer_indices) as cap:
        while step < args.steps:
            ids = next(train_stream).to(args.teacher_device)
            with torch.inference_mode():
                teacher(input_ids=ids, attention_mask=torch.ones_like(ids))

            # One teacher forward feeds several student steps.
            n_tokens = args.batch_size * args.seq_len
            n_chunks = max(1, math.ceil(n_tokens / args.student_batch_tokens))

            for li in layer_indices:
                xb_all = cap.x[li].to(args.student_device, torch.float32)
                yb_all = cap.y[li].to(args.student_device, torch.float32)
                for c in range(n_chunks):
                    lo = c * args.student_batch_tokens
                    hi = min(lo + args.student_batch_tokens, n_tokens)
                    if lo >= hi:
                        break
                    xb, yb = xb_all[lo:hi], yb_all[lo:hi]
                    optimizers[li].zero_grad(set_to_none=True)
                    loss = F.mse_loss(students[li](xb), yb)
                    loss = loss / yb.detach().pow(2).mean().clamp(min=1e-6)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        students[li].parameters(), args.grad_clip)
                    optimizers[li].step()
                    schedulers[li].step()
                    running[li] += loss.item()

            step += n_chunks

            if step % args.log_interval < n_chunks:
                elapsed = time.time() - t0
                parts = " ".join(
                    f"L{li}={running[li] / max(1, args.log_interval):.5f}"
                    for li in layer_indices)
                print(f"  step {step:7d}/{args.steps} | {parts} | "
                      f"{elapsed:.0f}s")
                running = {li: 0.0 for li in layer_indices}

            if step % args.eval_interval < n_chunks:
                parts = " ".join(f"L{li}={evaluate(li):.5f}"
                                 for li in layer_indices)
                print(f"  step {step:7d} | rel_mse {parts}")

            if args.save_interval and step % args.save_interval < n_chunks:
                for li in layer_indices:
                    save(li)
                print(f"  step {step:7d} | checkpointed")

    print("\nFinal relative MSE:")
    for li in layer_indices:
        print(f"  layer {li:2d}: {evaluate(li):.6f}")
        save(li)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
