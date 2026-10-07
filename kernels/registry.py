#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Which kernel implementation is used for each (model, batch size).

DLR-Net is served with different kernels at different batch sizes — this is a
normal latency-vs-batch trade-off, not a special case. Small batches favour a
single cooperative kernel; large batches favour a wider parallel strategy or a
cuBLAS-based orchestration. ``resolve(model, B)`` returns the entry for a cell.

Each entry lists the callable and its recommended configuration. The boundaries
were set by measurement on H100, so they are specific to that GPU; re-run
``bench/bench_paper.py`` before trusting them on other hardware.
"""

REGISTRY = {
    "Qwen3-0.6B": [
        {
            "B_lo": 1, "B_hi": 4096,
            "kernel": "triton_fused",
            "callable": "kernels.triton_fused.dlrnet_fused",
            "config": "block_b autotuned over {1,2,4,8,16,32,64}, num_warps=4, num_stages=2",
            "note": "Single-launch fused kernel; small and medium batch.",
        },
        {
            "B_lo": 8192, "B_hi": 131072,
            "kernel": "wgmma_fused",
            "callable": "kernels.wgmma_fused.wgmma_fused_forward",
            "config": "D=1024, R=32, BB=64 (compile-time)",
            "note": "WGMMA + TMA kernel; large batch. Crosses over from "
                    "triton_fused around B=8192.",
        },
    ],
    "Qwen3-1.7B": [
        {
            "B_lo": 1, "B_hi": 2048,
            "kernel": "cooperative",
            "callable": "kernels.cooperative.cooperative_forward",
            "config": "per-batch (BLOCK_B, NP, BLOCK_K_V, BLOCK_N_U); see the wrapper",
            "note": "Cooperative NP-CTA kernel; small and medium batch.",
        },
        {
            "B_lo": 4096, "B_hi": 8192,
            "kernel": "monolithic",
            "callable": "kernels.monolithic.monolithic_forward",
            "config": "BLOCK_B=32, BLOCK_K_V=128, BLOCK_N_U=128, NWARPS=8 (compile-time)",
            "note": "Monolithic fused prefill; wins at B=4096..8192. "
                    "(wgmma_persistent ties at 8192 but loses at 4096.)",
        },
        {
            "B_lo": 16384, "B_hi": 131072,
            "kernel": "wgmma_persistent",
            "callable": "kernels.wgmma_persistent.wgmma_persistent_forward",
            "config": "D=2048, R=64, BB=64 (compile-time)",
            "note": "WGMMA + persistent balanced-wave grid; wins at B>=16384 "
                    "(~5.2-5.4x SwiGLU vs monolithic's ~5.7x). The Triton tiled "
                    "kernel is 1.5-2x slower here at default config and is not used.",
        },
    ],
    "Qwen3-8B": [
        {
            "B_lo": 1, "B_hi": 8,
            "kernel": "cooperative_decode_8b",
            "callable": "kernels.cooperative_decode_8b.cooperative_decode_8b_forward",
            "config": "per-batch (NP, NWARPS, BLOCK_K); see the wrapper",
            "note": "Cooperative decode kernel; single-token / tiny batch.",
        },
        {
            "B_lo": 16, "B_hi": 1024,
            "kernel": "cooperative_8b",
            "callable": "kernels.cooperative_8b.cooperative_8b_forward",
            "config": "per-batch (BLOCK_B, NP, BLOCK_K_V, BLOCK_N_U); see the wrapper",
            "note": "Cooperative prefill kernel; small-to-moderate batch. Single "
                    "fused launch avoids the orchestrator's ~430-kernel launch "
                    "floor, which is why it wins to B=1024 (5.5x at B=128 vs the "
                    "orchestrator's 9x). NP tuned per batch for a 1-wave fit.",
        },
        {
            "B_lo": 2048, "B_hi": 4096,
            "kernel": "orchestrated",
            "callable": "kernels.orchestrated.make_orchestrated_graph",
            "config": "BLOCK_R=128, BLOCK_D=1024, ROWS_PER_PROG=2, num_warps=1, num_stages=2",
            "note": "cuBLAS + fused Triton epilogue + CUDA graph; wins once the "
                    "matmuls are large enough to amortize (B>=2048).",
        },
        {
            "B_lo": 8192, "B_hi": 131072,
            "kernel": "orchestrated_chunked",
            "callable": "kernels.orchestrated.make_orchestrated_chunked_graph",
            "config": "b_chunk=4096, BLOCK_R=128, BLOCK_D=1024, ROWS_PER_PROG=2, num_warps=1, num_stages=2",
            "note": "Chunked orchestration (per-chunk hidden state stays L2-resident); large batch.",
        },
    ],
    "Qwen3-32B": [
        {
            "B_lo": 1, "B_hi": 8,
            "kernel": "cooperative_32b",
            "callable": "kernels.cooperative_32b.cooperative_32b_forward",
            "config": "NP=20, BLOCK_B=16, BLOCK_K_V=64, BLOCK_N_U=64, NWARPS=8",
            "note": "Cooperative prefill kernel; single-token / tiny batch "
                    "(B=1 direct, B=2..8 padded to BLOCK_B).",
        },
        {
            "B_lo": 16, "B_hi": 4096,
            "kernel": "orchestrated",
            "callable": "kernels.orchestrated.make_orchestrated_graph",
            "config": "BLOCK_R=256, BLOCK_D=1024, ROWS_PER_PROG=2, num_warps=1, num_stages=3",
            "note": "cuBLAS + fused Triton epilogue + CUDA graph; small to intermediate batch.",
        },
        {
            "B_lo": 8192, "B_hi": 32768,
            "kernel": "orchestrated_chunked",
            "callable": "kernels.orchestrated.make_orchestrated_chunked_graph",
            "config": "b_chunk=4096, BLOCK_R=256, BLOCK_D=1024, ROWS_PER_PROG=2, num_warps=1, num_stages=3",
            "note": "Chunked orchestration; large batch up to the chunk-count crossover.",
        },
        {
            "B_lo": 65536, "B_hi": 131072,
            "kernel": "orchestrated",
            "callable": "kernels.orchestrated.make_orchestrated_graph",
            "config": "BLOCK_R=256, BLOCK_D=1024, ROWS_PER_PROG=2, num_warps=1, num_stages=3",
            "note": "Non-chunked orchestration wins at the largest batch, where the "
                    "per-chunk overhead outweighs the L2 benefit of chunking.",
        },
    ],
}


def resolve(model: str, batch_size: int):
    """Return the kernel entry for a given (model, B), or None if uncovered."""
    for entry in REGISTRY.get(model, []):
        if entry["B_lo"] <= batch_size <= entry["B_hi"]:
            return entry
    return None
