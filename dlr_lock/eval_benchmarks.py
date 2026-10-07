#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Quality preservation: does the locked model still behave like the original?

Evaluates the pretrained model and its locked counterpart side by side on

* perplexity over a held-out corpus, and
* downstream accuracy via the LM Evaluation Harness (MMLU, ARC Easy and
  Challenge, HellaSwag, WinoGrande, BoolQ, PIQA).

Locking is only a defense if legitimate users are unaffected, so this is the
first thing to check after distillation. Perplexity on the distillation corpus
recovers almost completely; the downstream benchmarks are the stricter test.

Note the metric conventions used in the paper: ``acc_norm`` for ARC Easy and
Challenge, HellaSwag, and PIQA (length-normalized, the standard for these
multiple-choice sets), and plain ``acc`` for WinoGrande and BoolQ.

Example
-------
    python -m dlr_lock.eval_benchmarks \\
        --model Qwen/Qwen3-0.6B --rank 32 \\
        --locked-dir runs/qwen3-0.6b/phase2 \\
        --output results/qwen3-0.6b-quality.json
"""

import argparse
import json
import math
import os

import torch

from .data import build_text_stream
from .loading import (autocast, empty_cache, load_causal_lm,
                      load_tokenizer)
from .model import build_locked_model, load_dlrnet_weights

DEFAULT_TASKS = ("mmlu,arc_easy,arc_challenge,hellaswag,winogrande,"
                 "boolq,piqa")

# Which lm-eval metric to report per task.
METRIC_BY_TASK = {
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "mmlu": "acc",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--locked-dir", required=True,
                   help="Directory of DLR-Net weights (Phase 1 or Phase 2).")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--multiplier", type=float, default=1.0)

    p.add_argument("--dataset", default="wikitext", choices=["wikitext", "c4"])
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=50)

    p.add_argument("--tasks", default=DEFAULT_TASKS,
                   help="Comma-separated lm-eval tasks, or '' to skip. "
                        "Generative tasks such as gsm8k are far slower than "
                        "the multiple-choice ones.")
    p.add_argument("--lm-eval-batch-size", type=int, default=16)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", default=None,
                   help="Pin the Hugging Face Hub revision (branch, tag, or "
                        "commit hash) for the model and dataset. Unset means "
                        "the default branch, whose contents can change; pin it "
                        "for reproducible runs.")
    p.add_argument("--output", default="")

    return p.parse_args(argv)


@torch.no_grad()
def evaluate_ppl(model, tokenizer, dataset, seq_len, batch_size, n_steps,
                 device, seed, revision=None):
    stream = build_text_stream(tokenizer, dataset, "validation", seq_len,
                               batch_size, seed, revision=revision)
    model.eval()
    total = 0.0
    with autocast(device):
        for _ in range(n_steps):
            ids = next(stream).to(device)
            total += model(input_ids=ids, labels=ids).loss.item()
    return math.exp(total / max(n_steps, 1))


def run_lm_eval(model, tokenizer, tasks, batch_size):
    """Run the LM Evaluation Harness; returns ``{task: {metric: value}}``."""
    if not tasks:
        return {}
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("  lm_eval not installed; skipping benchmarks "
              "(pip install lm-eval).")
        return {}

    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              batch_size=batch_size, dtype="bfloat16")
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks,
                                      batch_size=batch_size)

    scores = {}
    for task, metrics in results.get("results", {}).items():
        scores[task] = {}
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            scores[task][metric.replace(",none", "")] = value
    return scores


def collect(model, tokenizer, args, label):
    print(f"\n--- {label} ---")
    out = {}

    ppl = evaluate_ppl(model, tokenizer, args.dataset, args.seq_len,
                       args.eval_batch_size, args.eval_steps, args.device,
                       args.seed, revision=args.revision)
    out[f"{args.dataset}_ppl"] = ppl
    print(f"  {args.dataset} PPL: {ppl:.2f}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if tasks:
        print(f"  lm-eval: {', '.join(tasks)} ...")
        for task, metrics in run_lm_eval(
                model, tokenizer, tasks, args.lm_eval_batch_size).items():
            preferred = METRIC_BY_TASK.get(task)
            if preferred and preferred in metrics:
                out[f"{task}/{preferred}"] = metrics[preferred]
                print(f"    {task}/{preferred}: {metrics[preferred]:.4f}")
            else:
                for metric, value in metrics.items():
                    out[f"{task}/{metric}"] = value
    return out


def print_comparison(original, locked):
    print("\n" + "=" * 64)
    print(f"  {'Metric':<30s} {'Original':>10s} {'Locked':>10s} {'Delta':>10s}")
    print("-" * 64)
    for key, o in original.items():
        l = locked.get(key)
        if o is None or l is None:
            continue
        delta = l - o
        print(f"  {key:<30s} {o:>10.4f} {l:>10.4f} {delta:>+10.4f}")
    print("=" * 64)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.model, revision=args.revision)
    device_map = args.device_map or None

    original_model = load_causal_lm(args.model, device=args.device,
                                    device_map=device_map,
                                    revision=args.revision)
    original = collect(original_model, tokenizer, args, "Pretrained")
    del original_model
    empty_cache(args.device)

    locked_model = build_locked_model(
        args.model, args.rank, args.multiplier, device=args.device,
        device_map=device_map, torch_dtype=torch.bfloat16,
        revision=args.revision)
    load_dlrnet_weights(locked_model, args.locked_dir)
    locked = collect(locked_model, tokenizer, args, "Locked (DLR-Lock)")

    print_comparison(original, locked)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "locked_dir": args.locked_dir,
                       "rank": args.rank, "original": original,
                       "locked": locked}, f, indent=2)
        print(f"\nWrote {args.output}")

    return original, locked


if __name__ == "__main__":
    main()
