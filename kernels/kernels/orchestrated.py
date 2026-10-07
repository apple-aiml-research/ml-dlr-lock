#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Orchestrated DLR-Net forward: cuBLAS matmuls + one fused Triton epilogue,
captured in a CUDA graph — for intermediate and large batch on the larger models.

Instead of a single custom kernel, each layer is run as three steps:

    z = h @ Vᵀ                 # cuBLAS matmul (V projection)
    z = silu(z * rstd(h))      # one small Triton kernel (RMSNorm scale + SiLU)
    h = h + z @ U              # cuBLAS matmul + residual (addmm_)

The whole L-layer loop is captured once in a CUDA graph so per-launch overhead is
paid only at capture time. This wins on the larger models (8B, 32B) where the
per-layer matmuls are big enough for cuBLAS to amortize; the smaller models are
better served by the single-kernel implementations.

Two variants:
  - ``make_orchestrated_graph``: one chunk covering the full batch.
  - ``make_orchestrated_chunked_graph(b_chunk=4096)``: splits the batch into chunks so
    each chunk's hidden state stays resident in L2, which helps at large batch
    where the full-batch hidden state would evict itself between the V read and
    the U write. (At very large chunk counts the per-chunk overhead outweighs the
    L2 benefit, so the non-chunked variant is preferred there.)

Weights come from ``reference.pack_dlrnet_weights`` (Vs and Us shape (L, R, D),
premultiplied). Suggested epilogue config: BLOCK_R = R, BLOCK_D = 1024,
ROWS_PER_PROG = 2, num_warps = 1, num_stages = 2 (use num_stages = 3 at R = 256).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rstd_silu_kernel(
    H_ptr, Z_ptr, B: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    INV_D: tl.constexpr, EPS: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, ROWS_PER_PROG: tl.constexpr,
):
    """Fused (rstd of h) → (z * rstd) → silu → bf16 cast, in-place on Z.

    One program handles ROWS_PER_PROG rows. For each row:
      1. Stream H over the D axis (BLOCK_D at a time), accumulate the
         sum-of-squares, and form rstd.
      2. Stream Z over the R axis, apply z = silu(z * rstd), store back as bf16.
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    for ridx in tl.static_range(ROWS_PER_PROG):
        row = row_start + ridx
        if row < B:
            sq = 0.0
            for d_start in range(0, D, BLOCK_D):
                d_off = d_start + tl.arange(0, BLOCK_D)
                mask = d_off < D
                h = tl.load(H_ptr + row * D + d_off, mask=mask, other=0.0).to(tl.float32)
                sq += tl.sum(h * h)
            rstd = 1.0 / tl.sqrt(sq * INV_D + EPS)
            for r_start in range(0, R, BLOCK_R):
                r_off = r_start + tl.arange(0, BLOCK_R)
                mask = r_off < R
                z = tl.load(Z_ptr + row * R + r_off, mask=mask, other=0.0).to(tl.float32)
                z = z * rstd
                s = z / (1.0 + tl.exp(-z))
                tl.store(Z_ptr + row * R + r_off, s.to(tl.bfloat16), mask=mask)


def _run_fused(h, z, B, D, R, inv_d,
               block_r=128, block_d=1024, rows_per_prog=2, num_warps=1, num_stages=2):
    """Launch the fused RMSNorm-scale + SiLU epilogue kernel."""
    grid = ((B + rows_per_prog - 1) // rows_per_prog,)
    _fused_rstd_silu_kernel[grid](
        h, z, B, D, R, inv_d, 1e-6,
        block_r, block_d, rows_per_prog,
        num_warps=num_warps, num_stages=num_stages,
    )


def _capture_graph(step_fn, warmup=3):
    """Standard CUDA-graph capture with side-stream warmup."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            step_fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step_fn()
    return g


def make_orchestrated_graph(x, weights, block_r=128, block_d=1024,
                       rows_per_prog=2, num_warps=1, num_stages=2):
    """Non-chunked orchestrator, wrapped in one CUDA graph.

    Args:
      x: (B, D) bf16 input activations.
      weights: dict from pack_dlrnet_weights (Vs, Us shape (L, R, D)).
      block_r/block_d/rows_per_prog/num_warps/num_stages: epilogue config.

    Returns:
      (graph, h_buf) — call graph.replay(); the output is left in h_buf.
    """
    B, D = x.shape
    L, R = weights["L"], weights["R"]
    assert D == weights["D"]

    inv_d = 1.0 / D
    h_buf = x.clone()
    z_buf = torch.empty(B, R, device=x.device, dtype=x.dtype)
    Vs = weights["Vs"]   # (L, D, R) — already the layout h @ Vs[l] needs
    Us = weights["Us"]   # (L, R, D)

    def step():
        h_buf.copy_(x)
        for l in range(L):
            torch.matmul(h_buf, Vs[l], out=z_buf)
            _run_fused(h_buf, z_buf, B, D, R, inv_d,
                       block_r, block_d, rows_per_prog, num_warps, num_stages)
            h_buf.addmm_(z_buf, Us[l])

    g = _capture_graph(step)
    return g, h_buf


def make_orchestrated_chunked_graph(x, weights, b_chunk=4096,
                               block_r=128, block_d=1024,
                               rows_per_prog=2, num_warps=1, num_stages=2):
    """Chunked orchestrator: split B into b_chunk slices so each chunk's hidden
    state stays resident in L2. Helps at large batch; prefer the non-chunked
    variant once the chunk count is high.

    Args:
      x: (B, D) bf16. B must be divisible by b_chunk.
      weights: dict from pack_dlrnet_weights.
      b_chunk: rows per chunk.
      block_r/block_d/rows_per_prog/num_warps/num_stages: epilogue config.

    Returns:
      (graph, h_buf) — call graph.replay(); the output is left in h_buf.
    """
    B, D = x.shape
    L, R = weights["L"], weights["R"]
    assert D == weights["D"]
    assert B % b_chunk == 0, f"B={B} must be divisible by b_chunk={b_chunk}"

    n_chunks = B // b_chunk
    inv_d = 1.0 / D
    h_buf = x.clone()
    z_buf = torch.empty(b_chunk, R, device=x.device, dtype=x.dtype)
    Vs = weights["Vs"]   # (L, D, R)
    Us = weights["Us"]   # (L, R, D)

    def step():
        h_buf.copy_(x)
        for c in range(n_chunks):
            h_view = h_buf[c * b_chunk : (c + 1) * b_chunk]
            for l in range(L):
                torch.matmul(h_view, Vs[l], out=z_buf)
                _run_fused(h_view, z_buf, b_chunk, D, R, inv_d,
                           block_r, block_d, rows_per_prog, num_warps, num_stages)
                h_view.addmm_(z_buf, Us[l])

    g = _capture_graph(step)
    return g, h_buf
