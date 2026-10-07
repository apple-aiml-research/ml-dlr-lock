#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Monolithic fused DLR-Net prefill kernel — 1.7B (D=2048, R=64), moderate batch.

One CTA processes a tile of BLOCK_B rows and keeps that tile's hidden state
resident in shared memory across all layers. Each layer does RMSNorm, a K-tiled V
matmul, SiLU, then an N-tiled U matmul, with cp.async double-buffered weight
loads. The low-rank weights are streamed in K/N chunks (BLOCK_K_V / BLOCK_N_U
wide) because they do not all fit in shared memory at once.

Requires B to be a multiple of BLOCK_B.
"""
import torch
from ._cuda_loader import load_kernel

# Tile configuration (one CTA per BLOCK_B rows).
_BLOCK_B, _BLOCK_K_V, _BLOCK_N_U, _NWARPS = 32, 128, 128, 8


def monolithic_forward(x, Vs, Us):
    """Forward pass for 1.7B DLR-Net at moderate batch (B a multiple of 32).

    Args:
        x:  (B, 2048) bf16 on CUDA. B must be divisible by 32.
        Vs: (L, 64, 2048) bf16, premultiplied.
        Us: (L, 64, 2048) bf16, premultiplied.

    Returns: (B, 2048) bf16.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.shape[1] == 2048, f"monolithic kernel is D=2048 (got {x.shape[1]})"
    assert Vs.shape[1] == 64, f"monolithic kernel is R=64 (got {tuple(Vs.shape)})"
    assert x.shape[0] % _BLOCK_B == 0, f"B must be divisible by {_BLOCK_B} (got {x.shape[0]})"

    mod = load_kernel("monolithic.cu", "dlrnet_monolithic", {
        "CFG_D": 2048, "CFG_R": 64, "CFG_BLOCK_B": _BLOCK_B,
        "CFG_BLOCK_K_V": _BLOCK_K_V, "CFG_BLOCK_N_U": _BLOCK_N_U, "CFG_NWARPS": _NWARPS,
    })
    return mod.dlrnet_monolithic(x, Vs, Us)
