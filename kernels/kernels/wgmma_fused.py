#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fused DLR-Net prefill kernel with WGMMA + TMA — 0.6B (D=1024, R=32), large batch.

Uses Hopper tensor-core features: warpgroup-async matmuls (WGMMA) for both the V
and U projections, the Tensor Memory Accelerator (TMA) to load weight tiles, and
warpgroup specialization so one warpgroup runs a matmul while another performs the
RMSNorm reduction or issues the next TMA load. The residual add is folded into the
U-matmul accumulator, and the hidden state stays resident in shared memory across
all layers.

The kernel processes BLOCK_B rows per CTA and requires B to be a multiple of
BLOCK_B.
"""
import torch
from ._cuda_loader import load_kernel


def wgmma_fused_forward(x, Vs, Us):
    """Forward pass for 0.6B DLR-Net at large batch (B a multiple of 64).

    Args:
        x:  (B, 1024) bf16 on CUDA. B must be divisible by 64.
        Vs: (L, 32, 1024) bf16, premultiplied.
        Us: (L, 32, 1024) bf16, premultiplied.

    Returns: (B, 1024) bf16.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.shape[1] == 1024, f"wgmma_fused kernel is D=1024 (got {x.shape[1]})"
    assert Vs.shape[1] == 32, f"wgmma_fused kernel is R=32 (got {tuple(Vs.shape)})"
    assert x.shape[0] % 64 == 0, f"B must be divisible by 64 (got {x.shape[0]})"

    mod = load_kernel("wgmma_fused.cu", "dlrnet_wgmma_fused",
                      {"CFG_D": 1024, "CFG_R": 32, "CFG_BB": 64})
    return mod.dlrnet_wgmma_fused(x, Vs, Us)
