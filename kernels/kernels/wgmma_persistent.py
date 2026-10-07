#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fused DLR-Net prefill kernel with WGMMA + persistent balanced-wave grid —
1.7B (D=2048, R=64), large batch.

Uses warpgroup-async matmuls (WGMMA) for the V and U projections with
double-buffered TMA weight loads and warpgroup specialization. Rather than
launching one CTA per row-tile — which would leave a partial final wave of CTAs
that under-uses memory bandwidth — it launches a fixed number of persistent CTAs
sized from an occupancy query, so the work splits into the minimum number of
equally-full hardware waves; each CTA then strides over several row-tiles. Keeping
every wave full sustains memory bandwidth across batch sizes.

Requires B to be a multiple of BLOCK_B.
"""
import torch
from ._cuda_loader import load_kernel


def wgmma_persistent_forward(x, Vs, Us):
    """Forward pass for 1.7B DLR-Net at large batch (B a multiple of 64).

    Args:
        x:  (B, 2048) bf16 on CUDA. B must be divisible by 64.
        Vs: (L, 64, 2048) bf16, premultiplied.
        Us: (L, 64, 2048) bf16, premultiplied.

    Returns: (B, 2048) bf16.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.shape[1] == 2048, f"wgmma_persistent kernel is D=2048 (got {x.shape[1]})"
    assert Vs.shape[1] == 64, f"wgmma_persistent kernel is R=64 (got {tuple(Vs.shape)})"
    assert x.shape[0] % 64 == 0, f"B must be divisible by 64 (got {x.shape[0]})"

    mod = load_kernel("wgmma_persistent.cu", "dlrnet_wgmma_persistent",
                      {"CFG_D": 2048, "CFG_R": 64, "CFG_BB": 64})
    return mod.dlrnet_wgmma_persistent(x, Vs, Us)
