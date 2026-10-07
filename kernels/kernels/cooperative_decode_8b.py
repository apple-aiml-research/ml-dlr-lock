#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cooperative DLR-Net decode kernel — 8B (D=4096, R=128), batch B = 1..8.

A small-batch variant of the cooperative design for the decode regime. The grid
is (NP, B): each of the B rows gets its own group of NP cooperating CTAs, with
per-row barriers and reduction buffers. The replicated weights are served from
L2. The next layer's U tiles are staged through shared memory with cp.async.

The number of cooperating CTAs per row is chosen by batch size so the launch
stays within one wave.
"""
import torch
from ._cuda_loader import load_kernel

# (B_lo, B_hi, NP, NWARPS, BLOCK_K)
_CFG = [
    (1, 4, 32, 4, 512),
    (8, 8, 16, 8, 512),
]


def _resolve_cfg(B):
    for (lo, hi, np_, nw, bk) in _CFG:
        if lo <= B <= hi:
            return np_, nw, bk
    raise ValueError(f"No cooperative-decode config for B={B}; add a row to _CFG.")


def cooperative_decode_8b_forward(x, Vs, Us):
    """Forward pass for 8B DLR-Net at B ∈ {1, 2, 4, 8}.

    Args:
        x:  (B, 4096) bf16 on CUDA.
        Vs: (L, 128, 4096) bf16, premultiplied.
        Us: (L, 128, 4096) bf16, premultiplied.

    Returns: (B, 4096) bf16.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.shape[1] == 4096, f"cooperative_decode_8b kernel is D=4096 (got {x.shape[1]})"
    assert Vs.shape[1] == 128, f"cooperative_decode_8b kernel is R=128 (got {tuple(Vs.shape)})"

    np_, nw, bk = _resolve_cfg(x.shape[0])
    mod = load_kernel("cooperative_decode_8b.cu", "dlrnet_cooperative_decode_8b", {
        "CFG_D": 4096, "CFG_R": 128, "CFG_NP": np_,
        "CFG_NWARPS": nw, "CFG_BLOCK_K": bk, "CFG_FENCE": 1,
    })
    return mod.dlrnet_cooperative_decode_8b(x, Vs, Us)
