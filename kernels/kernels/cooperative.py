#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cooperative fused DLR-Net prefill kernel — 1.7B (D=2048, R=64), batch B ≤ 2048.

A group of NP CTAs cooperate on one tile of BLOCK_B rows. Each CTA owns a 1/NP
stripe of the model dimension: it computes the V matmul over its own slice of the
contraction axis and the U matmul over the matching output slice, so the two
slices coincide and no per-layer exchange of the hidden state between CTAs is
needed. A once-per-layer barrier plus a small global reduction combine the
partial V-projection and the RMSNorm sum-of-squares across the NP CTAs; the next
layer's weights are prefetched across the layer boundary.

The number of cooperating CTAs is chosen per batch size so the launch fills a
single wave of the GPU (grid = (B / BLOCK_B) * NP ≈ one CTA per SM). Small
batches use more CTAs per tile (more parallelism per row); larger batches use
fewer (the row tiles already supply parallelism).
"""
import torch
from ._cuda_loader import load_kernel

# (B_lo, B_hi, BLOCK_B, NP, BLOCK_K_V, BLOCK_N_U) — one wave per launch.
_CFG = [
    (   1,    8, 16, 16, 128, 128),   # input is padded up to BLOCK_B rows
    (  16,   64, 16, 16, 128, 128),
    ( 128,  256, 16,  8, 128, 128),
    ( 512,  512, 16,  4, 256, 256),   # low NP -> wider tiles cut the chunk count (-16%)
    (1024, 1024, 16,  2, 256, 256),   # wider tiles reduce the chunk count at low NP
    (2048, 2048, 32,  2, 128, 128),
]


def _resolve_cfg(B):
    for (lo, hi, bb, np_, bk, bn) in _CFG:
        if lo <= B <= hi:
            return bb, np_, bk, bn
    raise ValueError(f"No cooperative-kernel config for B={B}; add a row to _CFG.")


def cooperative_forward(x, Vs, Us):
    """Forward pass for 1.7B DLR-Net at B ∈ {1, 2, 4, ..., 2048}.

    Args:
        x:  (B, 2048) bf16 on CUDA.
        Vs: (L, 64, 2048) bf16, premultiplied (RMSNorm gain folded in).
        Us: (L, 64, 2048) bf16, premultiplied (residual scale folded in).

    Returns: (B, 2048) bf16. When B is below BLOCK_B the input is zero-padded;
    per-CTA cost is fixed by BLOCK_B, so padded rows cost nothing.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.shape[1] == 2048, f"cooperative kernel is D=2048 (got {x.shape[1]})"
    assert Vs.shape[1] == 64, f"cooperative kernel is R=64 (got {tuple(Vs.shape)})"

    B = x.shape[0]
    bb, np_, bk, bn = _resolve_cfg(B)
    mod = load_kernel("cooperative.cu", "dlrnet_cooperative", {
        "CFG_D": 2048, "CFG_R": 64, "CFG_BLOCK_B": bb,
        "CFG_BLOCK_K_V": bk, "CFG_BLOCK_N_U": bn, "CFG_NP": np_, "CFG_FENCE": 0,
    })

    if B % bb != 0:
        pad = bb - (B % bb)
        x_pad = torch.zeros((B + pad, x.shape[1]), device=x.device, dtype=x.dtype)
        x_pad[:B].copy_(x)
        return mod.dlrnet_cooperative(x_pad, Vs, Us)[:B]
    return mod.dlrnet_cooperative(x, Vs, Us)
