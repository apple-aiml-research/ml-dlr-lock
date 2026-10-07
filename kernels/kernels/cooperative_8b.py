#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cooperative fused DLR-Net prefill kernel — 8B (D=4096, R=128), batch B = 16..1024.

Same cooperative design as ``cooperative.py``: NP CTAs share one tile of BLOCK_B
rows, each owning a 1/NP stripe of the model dimension for both the V and U
matmuls, with a per-layer barrier and a small global reduction. At this scale the
hidden-state shared-memory buffer holds only each CTA's own stripe, which frees
enough shared memory for wider weight tiles.

NP is chosen per batch size so the launch fills a single wave (grid =
(B/BLOCK_B)·NP ≈ one CTA per SM): more cooperating CTAs at small batch, fewer as
the row tiles themselves supply parallelism.
"""
import torch
from ._cuda_loader import load_kernel

# (B_lo, B_hi, BLOCK_B, NP, BLOCK_K_V, BLOCK_N_U)
_CFG = [
    (  16,   64, 16, 16, 256, 128),
    ( 128,  256, 16,  8, 256, 128),
    ( 512,  512, 16,  4, 128, 128),
    (1024, 1024, 32,  4, 128, 128),
]


def _resolve_cfg(B):
    for (lo, hi, bb, np_, bk, bn) in _CFG:
        if lo <= B <= hi:
            return bb, np_, bk, bn
    raise ValueError(f"No cooperative_8b config for B={B}; add a row to _CFG.")


def cooperative_8b_forward(x, Vs, Us):
    """Forward pass for 8B DLR-Net at B in {16, 32, ..., 1024}.

    Args:
        x:  (B, 4096) bf16 on CUDA.
        Vs: (L, 128, 4096) bf16, premultiplied.
        Us: (L, 128, 4096) bf16, premultiplied.

    Returns: (B, 4096) bf16 (input zero-padded up to BLOCK_B; padded rows free).
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.shape[1] == 4096, f"cooperative_8b kernel is D=4096 (got {x.shape[1]})"
    assert Vs.shape[1] == 128, f"cooperative_8b kernel is R=128 (got {tuple(Vs.shape)})"

    B = x.shape[0]
    bb, np_, bk, bn = _resolve_cfg(B)
    mod = load_kernel("cooperative_8b.cu", "dlrnet_cooperative_8b", {
        "CFG_D": 4096, "CFG_R": 128, "CFG_BLOCK_B": bb,
        "CFG_BLOCK_K_V": bk, "CFG_BLOCK_N_U": bn, "CFG_NP": np_, "CFG_FENCE": 0,
    })

    if B % bb != 0:
        pad = bb - (B % bb)
        x_pad = torch.zeros((B + pad, x.shape[1]), device=x.device, dtype=x.dtype)
        x_pad[:B].copy_(x)
        return mod.dlrnet_cooperative_8b(x_pad, Vs, Us)[:B]
    return mod.dlrnet_cooperative_8b(x, Vs, Us)
