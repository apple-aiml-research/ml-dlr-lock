#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fused single-kernel Triton implementation of the full DLR-Net forward pass.

One CTA loads a tile of BLOCK_B rows, keeps h in registers across all L layers,
does per-layer RMSNorm → V matmul → SiLU → U matmul → residual add. The whole
depth-L computation is a single Triton kernel launch — no cross-layer host
overhead, no HBM round-trip for h between layers.

Precision: h kept bf16 across layers (matches PyTorch DLR-Net + SwiGLU baseline).
Accumulators are fp32 (free — register-only casts).

Weights must be premultiplied (RMSNorm gain folded into V, alpha folded into U)
so the kernel does not load w or alpha per layer. See reference.pack_dlrnet_weights.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dlrnet_fused_kernel(
    X_ptr, O_ptr,
    Vs_ptr, Us_ptr,
    B,
    stride_Vs_l, stride_Us_l,
    L,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    EPS: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    b_off = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_off < B
    d_off = tl.arange(0, D)
    r_off = tl.arange(0, R)

    # h in bf16 across all L layers (matches SwiGLU baseline precision)
    h = tl.load(X_ptr + b_off[:, None] * D + d_off[None, :],
                mask=b_mask[:, None], other=0.0)  # bf16

    inv_D = 1.0 / D

    for l in tl.range(0, L, num_stages=LOOP_STAGES):
        # fp32 reduction for numerical stability (register-only cast)
        h_f = h.to(tl.float32)
        ms = tl.sum(h_f * h_f, axis=1, keep_dims=True) * inv_D
        rstd = 1.0 / tl.sqrt(ms + EPS)

        V = tl.load(Vs_ptr + l * stride_Vs_l + d_off[:, None] * R + r_off[None, :])
        z_prime = tl.dot(h, V, out_dtype=tl.float32)
        z = z_prime * rstd

        sz = (z * tl.sigmoid(z)).to(tl.bfloat16)

        U = tl.load(Us_ptr + l * stride_Us_l + r_off[:, None] * D + d_off[None, :])
        dh = tl.dot(sz, U, out_dtype=tl.float32)
        h = (h_f + dh).to(tl.bfloat16)

    tl.store(O_ptr + b_off[:, None] * D + d_off[None, :],
             h, mask=b_mask[:, None])


def dlrnet_fused(x, weights, block_b=8, num_warps=4, num_stages=2, out=None):
    """Run the fused DLR-Net forward on x. Weights must be premultiplied."""
    assert weights.get("premultiplied", False), \
        "dlrnet_fused requires premultiplied weights (RMSNorm gain into V, alpha into U)"
    B, D = x.shape
    L, R = weights["L"], weights["R"]
    assert D == weights["D"]
    if out is None:
        out = torch.empty_like(x)
    grid = (triton.cdiv(B, block_b),)
    _dlrnet_fused_kernel[grid](
        x, out,
        weights["Vs"], weights["Us"],
        B,
        weights["Vs"].stride(0),
        weights["Us"].stride(0),
        L,
        D=D, R=R,
        BLOCK_B=block_b,
        EPS=1e-6,
        LOOP_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
