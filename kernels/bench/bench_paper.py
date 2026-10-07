#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Unified benchmark for the DLR-Net forward kernels.

For every (model, batch size) cell, this resolves the chosen implementation from
``registry.py``, runs it with the clock-locked 5-pass methodology, checks its
output against the PyTorch DLR-Net reference, and reports latency relative to the
torch-compiled SwiGLU baseline. It doubles as the correctness regression check
for every kernel in ``kernels/``.

Usage:
    python bench/bench_paper.py                       # all models, default B sweep
    python bench/bench_paper.py --models Qwen3-1.7B   # one model
    python bench/bench_paper.py --candidates          # also run the 1.7B large-B
                                                      # alternatives head-to-head

Requires an H100 (sm_90a) with nvidia-smi clock-lock permission.
"""
import os
import sys
import json
import argparse
import importlib

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                     # kernels/
sys.path.insert(0, _PKG)
sys.path.insert(0, _HERE)                         # bench/ (for _protocol)

import registry
import _protocol
from _protocol import (N_WARMUP, N_ITERS, N_PASSES, lock_clocks, reset_clocks,
                       bench_pass, measure, free_all)
from reference import build_models, pack_dlrnet_weights, DTYPE

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

REL_ERR_TOL = 2e-2

DEFAULT_B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
             2048, 4096, 8192, 16384, 32768, 65536, 131072]

# 1.7B large-batch alternatives to compare head-to-head with --candidates.
CANDIDATES_17B = ["triton_tiled", "wgmma_persistent", "monolithic"]


# Timing harness (clock-locked, N-pass min-of-iters) lives in _protocol.py so
# bench_paper.py (single MLP) and bench_e2e.py (full model) share one method.


def _import_callable(path):
    mod_name, fn_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_name), fn_name)


# --------------------------------------------------------------------------- #
# Kernel dispatch: each kernel family has its own calling convention. Every
# builder returns a zero-argument run() that performs only the timed work and
# returns the output tensor (setup — weight transpose, graph capture, autotune —
# happens outside the timed region).
# --------------------------------------------------------------------------- #
_TRITON_WEIGHTS = {"triton_fused", "triton_tiled"}
_CU_XVU = {"wgmma_fused", "cooperative", "wgmma_persistent", "monolithic",
           "cooperative_decode_8b", "cooperative_8b", "cooperative_32b"}
_ORCH = {"orchestrated", "orchestrated_chunked"}


def build_run(kernel, callable_path, x, weights):
    """Return a zero-arg run() -> output tensor for the given kernel label."""
    R = weights["R"]

    if kernel == "triton_fused":
        fn = _import_callable(callable_path)
        best = None
        for bb in (1, 2, 4, 8, 16, 32, 64):
            try:
                fn(x, weights, block_b=bb, num_warps=4, num_stages=2)
                med = measure(lambda: fn(x, weights, block_b=bb, num_warps=4, num_stages=2))["mean_us"]
                if best is None or med < best[1]:
                    best = (bb, med)
            except Exception:
                pass
            free_all()
        bb = best[0] if best else 8
        return lambda: fn(x, weights, block_b=bb, num_warps=4, num_stages=2)

    if kernel == "triton_tiled":
        fn = _import_callable(callable_path)
        return lambda: fn(x, weights, block_b=32, block_k_v=128, block_n_u=128,
                          num_warps=8, num_stages=2)

    if kernel in _CU_XVU:
        fn = _import_callable(callable_path)
        Vs = weights["Vs"].transpose(-1, -2).contiguous()   # (L,D,R) -> (L,R,D)
        Us = weights["Us"]                                    # (L,R,D)
        return lambda: fn(x, Vs, Us)

    if kernel in _ORCH:
        make = _import_callable(callable_path)
        num_stages = 3 if R >= 256 else 2
        if kernel == "orchestrated_chunked":
            g, h_buf = make(x, weights, b_chunk=4096, block_r=R, block_d=1024,
                            rows_per_prog=2, num_warps=1, num_stages=num_stages)
        else:
            g, h_buf = make(x, weights, block_r=R, block_d=1024,
                            rows_per_prog=2, num_warps=1, num_stages=num_stages)

        def run():
            g.replay()
            return h_buf
        return run

    raise ValueError(f"Unknown kernel label: {kernel}")


# --------------------------------------------------------------------------- #
def rel_err(out, ref):
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def run_cell(model, B, kernel, callable_path, x, weights, ref, sw_mean):
    """Build, correctness-check, and time one (kernel, cell). Returns a dict."""
    try:
        run = build_run(kernel, callable_path, x, weights)
        out = run()
        err = rel_err(out, ref)
        if err > REL_ERR_TOL:
            return {"kernel": kernel, "status": "WRONG", "rel_err": err}
        stats = measure(run)
        stats.update({
            "kernel": kernel, "status": "ok", "rel_err": err,
            "ratio": stats["mean_us"] / sw_mean if sw_mean else None,
        })
        return stats
    except torch.cuda.OutOfMemoryError:
        free_all()
        return {"kernel": kernel, "status": "OOM"}
    except Exception as e:
        free_all()
        return {"kernel": kernel, "status": f"ERROR: {type(e).__name__}: {e}"}
    finally:
        free_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(registry.REGISTRY.keys()))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_B)
    ap.add_argument("--candidates", action="store_true",
                    help="also run the 1.7B large-B alternatives head-to-head")
    ap.add_argument("--passes", type=int, default=N_PASSES,
                    help="measurement passes (each = min over 100 iters); more passes -> tighter std")
    ap.add_argument("--out", default="/mnt/task_runtime/bench_paper.json")
    args = ap.parse_args()

    _protocol.N_PASSES = args.passes

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    locked = lock_clocks()
    results = {}

    try:
        for model in args.models:
            print(f"\n########## {model} ##########")
            swiglu, dlrnet, cfg = build_models(model)
            swiglu.eval(); dlrnet.eval()
            D = cfg["d"]
            weights = pack_dlrnet_weights(dlrnet, premultiply=True)
            sw_compiled = torch.compile(swiglu, mode="reduce-overhead")
            # Same off-the-shelf treatment applied to the DLR-Net architecture: the
            # 142-layer sequence captured into a CUDA graph. This is the honest
            # "no custom kernel" baseline the fused kernel is measured against.
            dl_compiled = torch.compile(dlrnet, mode="reduce-overhead")
            results[model] = {}

            for B in args.batch_sizes:
                free_all()
                try:
                    x = torch.randn(B, D, device="cuda", dtype=DTYPE)
                    with torch.no_grad():
                        ref = dlrnet(x)
                        sw = measure(lambda: sw_compiled(x))
                        sw_mean = sw["mean_us"]
                except torch.cuda.OutOfMemoryError:
                    print(f"  B={B:>6}: OOM building baseline"); free_all(); continue

                # reduce-overhead DLR-Net (compiled, no custom kernel).
                try:
                    with torch.no_grad():
                        dl_ro = measure(lambda: dl_compiled(x))
                    dl_ro_ratio = dl_ro["mean_us"] / sw_mean if sw_mean else None
                except torch.cuda.OutOfMemoryError:
                    dl_ro = {"status": "OOM"}; dl_ro_ratio = None; free_all()
                except Exception as e:
                    dl_ro = {"status": f"ERROR: {type(e).__name__}: {e}"}
                    dl_ro_ratio = None; free_all()

                entry = registry.resolve(model, B)
                cells = []
                if entry is not None:
                    cells.append((entry["kernel"], entry["callable"]))
                if args.candidates and model == "Qwen3-1.7B" and B >= 4096:
                    extra = {"triton_tiled": "kernels.triton_tiled.tiled_forward",
                             "wgmma_persistent": "kernels.wgmma_persistent.wgmma_persistent_forward",
                             "monolithic": "kernels.monolithic.monolithic_forward"}
                    cells = [(k, p) for k, p in extra.items()]

                row = {"SwiGLU": sw, "DLRNet_RO": dl_ro, "cells": []}
                line = f"  B={B:>6}: SwiGLU {sw_mean:>9.1f}±{sw['std_us']:.1f}us"
                if dl_ro.get("status") != "OOM" and "mean_us" in dl_ro:
                    line += (f" | DLR-RO {dl_ro['mean_us']:>9.1f}±{dl_ro['std_us']:.1f}us "
                             f"({dl_ro_ratio:.2f}x)")
                else:
                    line += f" | DLR-RO [{dl_ro.get('status', '?')}]"
                for kernel, path in cells:
                    r = run_cell(model, B, kernel, path, x, weights, ref, sw_mean)
                    row["cells"].append(r)
                    if r["status"] == "ok":
                        line += (f" | {kernel} {r['mean_us']:>9.1f}±{r['std_us']:.1f}us "
                                 f"({r['ratio']:.2f}x)")
                    else:
                        line += f" | {kernel} [{r['status']}]"
                print(line)
                results[model][B] = row
                del x, ref
                free_all()

        try:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved -> {args.out}")
        except Exception:
            pass
    finally:
        if locked:
            reset_clocks(); print("clocks reset.")


if __name__ == "__main__":
    main()
