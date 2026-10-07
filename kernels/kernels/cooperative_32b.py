#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cooperative fused DLR-Net prefill kernel — 32B (D=5120, R=256), batch B = 1..8.

The cooperative design from ``cooperative.py`` at the 32B scale: NP CTAs share a
tile of BLOCK_B rows, each owning a 1/NP stripe of the model dimension for both
matmuls, with a per-layer barrier and reduction. At this scale the hidden-state
shared-memory buffer holds only each CTA's own stripe and the cross-CTA reduction
is streamed rather than staged in full.

B = 1 runs directly; B ∈ {2, 4, 8} zero-pad up to BLOCK_B (per-CTA cost is fixed
by BLOCK_B, so the padded rows cost nothing).
"""
import torch
from ._cuda_loader import load_kernel

# Single configuration.
_BLOCK_B, _NP, _BLOCK_K_V, _BLOCK_N_U, _NWARPS = 16, 20, 64, 64, 8


def cooperative_32b_forward(x, Vs, Us):
    """Forward pass for 32B DLR-Net at B ∈ {1, 2, 4, 8}.

    Args:
        x:  (B, 5120) bf16 on CUDA.
        Vs: (L, 256, 5120) bf16, premultiplied.
        Us: (L, 256, 5120) bf16, premultiplied.

    Returns: (B, 5120) bf16.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.shape[1] == 5120, f"cooperative_32b kernel is D=5120 (got {x.shape[1]})"
    assert Vs.shape[1] == 256, f"cooperative_32b kernel is R=256 (got {tuple(Vs.shape)})"

    B = x.shape[0]
    mod = load_kernel("cooperative_32b.cu", "dlrnet_cooperative_32b", {
        "CFG_D": 5120, "CFG_R": 256, "CFG_BLOCK_B": _BLOCK_B,
        "CFG_BLOCK_K_V": _BLOCK_K_V, "CFG_BLOCK_N_U": _BLOCK_N_U,
        "CFG_NP": _NP, "CFG_NWARPS": _NWARPS, "CFG_FENCE": 0,
    })

    if B % _BLOCK_B != 0:
        pad = _BLOCK_B - (B % _BLOCK_B)
        x_pad = torch.zeros((B + pad, x.shape[1]), device=x.device, dtype=x.dtype)
        x_pad[:B].copy_(x)
        return mod.dlrnet_cooperative_32b(x_pad, Vs, Us)[:B]
    return mod.dlrnet_cooperative_32b(x, Vs, Us)
