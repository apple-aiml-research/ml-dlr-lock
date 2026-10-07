# kernels/

Kernel implementations for the DLR-Net forward pass, used for the latency figure.
DLR-Net replaces an MLP block with a stack of L low-rank residual layers; each
layer computes `h = h + alpha * U @ silu(V @ rmsnorm(h))` with V, U low-rank
(rank R << D). Weights are premultiplied (the RMSNorm gain is folded into V and
the residual scale into U) so the kernels never load them separately.

These are reasonable reference implementations, not maximally-tuned kernels. DLR-Net
is served with a different implementation depending on batch size — a routine
latency-vs-batch trade-off — dispatched by `registry.py`.

## Choosing an implementation

`registry.resolve(model, batch_size)` returns the entry (callable + recommended
config) for a cell. Broadly:

- **small / medium batch** → a single fused kernel that keeps the hidden state
  on-chip across all layers (`triton_fused`, `cooperative*`);
- **large batch** → a wider parallel strategy (`triton_tiled`, `wgmma_*`) or a
  cuBLAS-based orchestration (`orchestrated`).

## Status

| Model | B range | Implementation | File |
|---|---|---|---|
| Qwen3-0.6B | 1..4096 | `triton_fused` | `kernels/triton_fused.py` |
| Qwen3-0.6B | 8192..131072 | `wgmma_fused` | `kernels/cuda/wgmma_fused.cu` |
| Qwen3-1.7B | 1..2048 | `cooperative` | `kernels/cuda/cooperative.cu` |
| Qwen3-1.7B | 4096..8192 | `monolithic` | `kernels/cuda/monolithic.cu` |
| Qwen3-1.7B | 16384..131072 | `wgmma_persistent` | `kernels/cuda/wgmma_persistent.cu` |
| Qwen3-8B | 1..8 | `cooperative_decode_8b` | `kernels/cuda/cooperative_decode_8b.cu` |
| Qwen3-8B | 16..64 | `cooperative_8b` | `kernels/cuda/cooperative_8b.cu` |
| Qwen3-8B | 128..4096 | `orchestrated` | `kernels/orchestrated.py` |
| Qwen3-8B | 8192..131072 | `orchestrated` (chunked) | `kernels/orchestrated.py` |
| Qwen3-32B | 1..8 | `cooperative_32b` | `kernels/cuda/cooperative_32b.cu` |
| Qwen3-32B | 16..4096 | `orchestrated` | `kernels/orchestrated.py` |
| Qwen3-32B | 8192..32768 | `orchestrated` (chunked) | `kernels/orchestrated.py` |
| Qwen3-32B | 65536..131072 | `orchestrated` | `kernels/orchestrated.py` |

The 1.7B large-batch boundary was settled by a head-to-head comparison:
`monolithic` wins at B=4096–8192, `wgmma_persistent` at B≥16384. The Triton
tiled kernel (`kernels/triton_tiled.py`) is retained as an untuned alternative
but is 1.5–2× slower here, so it is not routed to.

## Directory layout

```
kernels/
├── README.md                 # this file
├── registry.py               # (model, batch size) -> which implementation
├── reference.py              # PyTorch DLR-Net + SwiGLU + pack_dlrnet_weights
└── kernels/
    ├── _cuda_loader.py           # shared JIT loader for the .cu kernels
    ├── triton_fused.py           # 0.6B small/medium — single-launch fused (Triton)
    ├── triton_tiled.py           # 1.7B large — tiled single-CTA (Triton)
    ├── orchestrated.py           # 8B/32B mid+large — cuBLAS + fused epilogue + CUDA graph
    └── cuda/
        ├── wgmma_fused.cu        # 0.6B large — WGMMA + TMA
        ├── cooperative.cu        # 1.7B small/medium — cooperative NP-CTA
        ├── wgmma_persistent.cu   # 1.7B large — WGMMA + persistent balanced-wave grid
        ├── monolithic.cu         # 1.7B — monolithic fused prefill
        ├── cooperative_8b.cu     # 8B small — cooperative (own-stripe hidden state)
        ├── cooperative_decode_8b.cu  # 8B tiny — cooperative decode
        └── cooperative_32b.cu    # 32B tiny — cooperative
    (plus cooperative*.py / wgmma_*.py / monolithic.py wrappers next to the Triton files)
```

Each `.cu` kernel has a thin Python wrapper in `kernels/` that selects the config
and calls the shared loader in `_cuda_loader.py`; the loader compiles each source
once per configuration via `torch.utils.cpp_extension`.

## Next steps

1. `bench/bench_paper.py`: one harness that imports `registry`, runs every
   (model, batch size) cell, and reports the latency ratio vs the SwiGLU baseline.
2. Settle the 1.7B large-batch boundary (`triton_tiled` vs `wgmma_persistent` vs
   `monolithic`), then drop the unused alternatives.
