#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared clock-locked timing protocol for the paper latency measurements.

Single source of truth for the methodology described in the paper's inference-
latency appendix: lock the GPU core and memory clocks with ``nvidia-smi`` (to
remove GPU-boost / DVFS run-to-run variance), then time each configuration as
the minimum over ``N_ITERS`` iterations (after ``N_WARMUP`` warmup iterations),
repeated for ``N_PASSES`` passes; report the mean across passes as the point
estimate with the across-pass standard deviation as the error bar.

Used by both ``bench_paper.py`` (single-MLP Figure-3 ratios) and
``bench_e2e.py`` (full-model prefill tables), so the two never drift apart.

``measure``/``bench_pass`` accept an optional ``pre_iter`` hook called once
per iteration inside the timed region — full-model reduce-overhead runs pass
``torch.compiler.cudagraph_mark_step_begin`` here so the CUDA-graph tree does
not reject cached tensors across iterations; single-MLP runs leave it ``None``.
"""
import gc
import statistics
import subprocess

import torch

N_WARMUP, N_ITERS, N_PASSES = 100, 100, 10


# --------------------------------------------------------------------------- #
# Clock lock (nvidia-smi). Requires an H100 (sm_90a) with lock permission.
# --------------------------------------------------------------------------- #
def _nvsmi_query(fields):
    return subprocess.check_output(
        ["nvidia-smi", "-i", "0", f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"], text=True).strip()


def lock_clocks():
    """Lock SM and (if the driver allows) memory clocks at their max. Returns
    True if the SM lock took hold, so the caller knows to reset afterwards."""
    try:
        sm, mem = (int(x) for x in _nvsmi_query("clocks.max.sm,clocks.max.memory").split(", "))
        subprocess.check_call(["nvidia-smi", "-i", "0", f"--lock-gpu-clocks={sm},{sm}"])

        # Memory-clock lock: some drivers reject the immediate form and require
        # the "deferred" variant. Try both; report which (if any) succeeded.
        mem_locked = False
        for label, cmd in (("immediate", ["nvidia-smi", "-i", "0", f"--lock-memory-clocks={mem},{mem}"]),
                           ("deferred",  ["nvidia-smi", "-i", "0", f"--lock-memory-clocks-deferred={mem}"])):
            try:
                subprocess.check_call(cmd)
                mem_locked = True
                print(f"  memory clock locked ({label}) at {mem} MHz")
                break
            except Exception as e:
                print(f"  memory clock lock ({label}) failed: {e}")

        # Report the ACTUAL clocks so the log shows whether the lock took hold.
        try:
            actual = _nvsmi_query("clocks.sm,clocks.mem")
            print(f"  target SM={sm} MEM={mem} MHz | actual SM,MEM now = {actual} MHz | mem_locked={mem_locked}")
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"  clock lock unavailable ({e}); timings will be noisier")
        return False


def reset_clocks():
    for cmd in ("--reset-gpu-clocks", "--reset-memory-clocks", "--reset-memory-clocks-deferred"):
        try:
            subprocess.check_call(["nvidia-smi", "-i", "0", cmd])
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Measurement.
# --------------------------------------------------------------------------- #
def bench_pass(fn, n=N_ITERS, warmup=N_WARMUP, pre_iter=None):
    """One measurement pass: return the list of n per-iteration times (us).

    ``pre_iter`` (if given) is invoked once per iteration inside the timed
    region, immediately before ``fn`` — used for cudagraph step markers.
    """
    for _ in range(warmup):
        if pre_iter is not None:
            pre_iter()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        if pre_iter is not None:
            pre_iter()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)   # ms -> microseconds
    return times


def measure(fn, passes=None, pre_iter=None, n_iters=None, n_warmup=None):
    """Clock-locked measurement following the paper method: each pass takes the
    min over n_iters iterations (rejects transient scheduling spikes); repeat for
    N_PASSES passes. Report the mean across passes as the point estimate.

    n_iters / n_warmup default to N_ITERS / N_WARMUP (the single-MLP figure
    protocol); full-model callers pass smaller counts since each forward is far
    longer. The measurement FORM is identical either way.

    Returns a dict with two notions of spread:
      - across-pass std (`std_us`): run-to-run reproducibility of the estimate
        (small under clock-lock — this is the "the number reproduces" bar);
      - within-pass per-iteration std (`iter_std_us`): raw call jitter over all
        passes' iterations (larger — the honest per-call spread).
    """
    if passes is None:
        passes = N_PASSES
    if n_iters is None:
        n_iters = N_ITERS
    if n_warmup is None:
        n_warmup = N_WARMUP
    pass_mins, all_iters = [], []
    for _ in range(passes):
        ts = bench_pass(fn, n=n_iters, warmup=n_warmup, pre_iter=pre_iter)
        pass_mins.append(min(ts))
        all_iters.extend(ts)
    mean = statistics.fmean(pass_mins)
    std = statistics.pstdev(pass_mins) if len(pass_mins) > 1 else 0.0
    median = sorted(pass_mins)[len(pass_mins) // 2]
    return {
        "mean_us": mean, "std_us": std, "median_us": median,
        "pass_mins_us": pass_mins,
        "iter_mean_us": statistics.fmean(all_iters),
        "iter_std_us": statistics.pstdev(all_iters),
    }


def free_all():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
