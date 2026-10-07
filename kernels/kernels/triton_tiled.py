#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tiled fused DLR-Net prefill kernel in Triton — for large batch.

One CTA processes a tile of BLOCK_B rows through all layers. Each layer streams
the low-rank weights in tiles: a K-tiled V matmul (BLOCK_K_V-wide chunks of the
contraction axis) then an N-tiled U matmul (BLOCK_N_U-wide chunks of the output).
The hidden state lives in a per-CTA scratch buffer that stays resident in L2 at
the wave counts these batch sizes produce. ``num_stages`` drives the software
pipelining of the weight loads.

Weights must be premultiplied (RMSNorm gain folded into V, residual scale folded
into U); see ``reference.pack_dlrnet_weights``. Recommended defaults: BLOCK_B=32,
BLOCK_K_V=128, BLOCK_N_U=128, num_warps=8, num_stages=2.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dlrnet_tiled_kernel(
    X_ptr, O_ptr, H_scratch_ptr,
    Vs_ptr, Us_ptr,
    B,
    stride_Vs_l, stride_Us_l,
    L,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K_V: tl.constexpr,
    BLOCK_N_U: tl.constexpr,
    NUM_K_V: tl.constexpr,
    NUM_N_U: tl.constexpr,
    EPS: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    """One CTA processes BLOCK_B rows through all L layers.

    H_scratch is (num_ctas * BLOCK_B, D) bf16 — each CTA keeps a persistent tile
    of the hidden state there across layers.
    """
    pid = tl.program_id(0)
    b_off = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_off < B

    r_off = tl.arange(0, R)

    h_base = pid * BLOCK_B
    inv_D = 1.0 / D

    # Copy this CTA's rows of X into the hidden-state scratch (full D).
    for kc in tl.range(0, NUM_K_V):
        k_off = kc * BLOCK_K_V + tl.arange(0, BLOCK_K_V)
        h_ck = tl.load(X_ptr + b_off[:, None] * D + k_off[None, :],
                       mask=b_mask[:, None], other=0.0)
        tl.store(H_scratch_ptr + (h_base + tl.arange(0, BLOCK_B))[:, None] * D + k_off[None, :],
                 h_ck)

    for l in tl.range(0, L, num_stages=1):
        # RMSNorm: streaming sum-of-squares over K-chunks of the hidden state.
        sq_acc = tl.zeros([BLOCK_B], tl.float32)
        for kc in tl.range(0, NUM_K_V, num_stages=LOOP_STAGES):
            k_off = kc * BLOCK_K_V + tl.arange(0, BLOCK_K_V)
            h_ck = tl.load(H_scratch_ptr +
                           (h_base + tl.arange(0, BLOCK_B))[:, None] * D + k_off[None, :])
            h_ck_f = h_ck.to(tl.float32)
            sq_acc += tl.sum(h_ck_f * h_ck_f, axis=1)
        rstd = 1.0 / tl.sqrt(sq_acc * inv_D + EPS)

        # V matmul, K-tiled: z = h @ V  (fp32 accumulator).
        z_acc = tl.zeros([BLOCK_B, R], tl.float32)
        for kc in tl.range(0, NUM_K_V, num_stages=LOOP_STAGES):
            k_off = kc * BLOCK_K_V + tl.arange(0, BLOCK_K_V)
            V_ck = tl.load(Vs_ptr + l * stride_Vs_l +
                           k_off[:, None] * R + r_off[None, :])
            h_ck = tl.load(H_scratch_ptr +
                           (h_base + tl.arange(0, BLOCK_B))[:, None] * D + k_off[None, :])
            z_acc += tl.dot(h_ck, V_ck, out_dtype=tl.float32)
        z = z_acc * rstd[:, None]
        sz = (z * tl.sigmoid(z)).to(tl.bfloat16)

        # U matmul, N-tiled: dh = sz @ U, added back into the hidden state.
        for nc in tl.range(0, NUM_N_U, num_stages=LOOP_STAGES):
            n_off = nc * BLOCK_N_U + tl.arange(0, BLOCK_N_U)
            U_ck = tl.load(Us_ptr + l * stride_Us_l +
                           r_off[:, None] * D + n_off[None, :])
            dh_ck = tl.dot(sz, U_ck, out_dtype=tl.float32)
            h_old = tl.load(H_scratch_ptr +
                            (h_base + tl.arange(0, BLOCK_B))[:, None] * D + n_off[None, :])
            h_new = (h_old.to(tl.float32) + dh_ck).to(tl.bfloat16)
            tl.store(H_scratch_ptr +
                     (h_base + tl.arange(0, BLOCK_B))[:, None] * D + n_off[None, :],
                     h_new)

    # Copy the hidden-state scratch back to the output.
    for kc in tl.range(0, NUM_K_V):
        k_off = kc * BLOCK_K_V + tl.arange(0, BLOCK_K_V)
        h_final = tl.load(H_scratch_ptr +
                          (h_base + tl.arange(0, BLOCK_B))[:, None] * D + k_off[None, :])
        tl.store(O_ptr + b_off[:, None] * D + k_off[None, :], h_final,
                 mask=b_mask[:, None])


def tiled_forward(x, weights, block_b=32, block_k_v=128, block_n_u=128,
                  num_warps=8, num_stages=2, out=None, h_scratch=None):
    """Run the tiled DLR-Net forward on x. D must divide both tile widths."""
    assert weights.get("premultiplied", False)
    B, D = x.shape
    L, R = weights["L"], weights["R"]
    assert D == weights["D"]
    assert D % block_k_v == 0
    assert D % block_n_u == 0

    num_k_v = D // block_k_v
    num_n_u = D // block_n_u

    grid_size = triton.cdiv(B, block_b)
    total_rows = grid_size * block_b

    if out is None:
        out = torch.empty_like(x)
    if h_scratch is None or h_scratch.shape[0] < total_rows or h_scratch.shape[1] < D:
        h_scratch = torch.empty(total_rows, D, device=x.device, dtype=x.dtype)

    _dlrnet_tiled_kernel[(grid_size,)](
        x, out, h_scratch,
        weights["Vs"], weights["Us"],
        B,
        weights["Vs"].stride(0),
        weights["Us"].stride(0),
        L,
        D=D, R=R,
        BLOCK_B=block_b,
        BLOCK_K_V=block_k_v,
        BLOCK_N_U=block_n_u,
        NUM_K_V=num_k_v,
        NUM_N_U=num_n_u,
        EPS=1e-6,
        LOOP_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
