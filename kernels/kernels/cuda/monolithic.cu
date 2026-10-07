//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Monolithic fused DLR-Net prefill kernel (D=2048, R=64).
//
// Each DLR-Net layer computes h += alpha * U @ silu(V @ rmsnorm(h)), with the
// RMS gain folded into V and alpha folded into U. One CTA processes a tile of
// BLOCK_B rows and keeps the whole h tile resident in shared memory across all
// L layers. Per layer it does RMSNorm, a K-tiled V matmul (mma.sync), SiLU, then
// an N-tiled U matmul, with cp.async double-buffered weight loads streamed in
// K/N chunks (BLOCK_K_V / BLOCK_N_U wide) since the low-rank weights don't fit
// in SMEM. At the target batch size the grid is a single wave over the SMs.
//
// Compile-time config: D=CFG_D, R=CFG_R, BLOCK_B=CFG_BLOCK_B,
// BLOCK_K_V=CFG_BLOCK_K_V, BLOCK_N_U=CFG_BLOCK_N_U, NWARPS=CFG_NWARPS.
// Requires B % BLOCK_B == 0.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;

#ifndef CFG_D
#define CFG_D 2048
#endif
#ifndef CFG_R
#define CFG_R 64
#endif

constexpr int D = CFG_D;
constexpr int R = CFG_R;
constexpr int BLOCK_B = 32;
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;
constexpr int BLOCK_K_V = 128;
constexpr int BLOCK_N_U = 128;
constexpr int NUM_K_V = D / BLOCK_K_V;
constexpr int NUM_N_U = D / BLOCK_N_U;
constexpr int K_TILE = 16;
constexpr int N_TILE = 8;
constexpr int K_TILES_PER_VCHUNK = BLOCK_K_V / K_TILE;
constexpr int K_TILES_U = R / K_TILE;
constexpr int N_TILES_PER_CHUNK = BLOCK_N_U / N_TILE;
constexpr int N_PER_WARP_U = N_TILES_PER_CHUNK / NWARPS;

__device__ __forceinline__ int sw128(int row, int col) {
    int tile_idx     = row >> 3;
    int row_in_tile  = row & 7;
    int core_col     = col >> 3;
    int sw_col       = core_col ^ row_in_tile;
    int elem         = col & 7;
    return tile_idx * 512 + row_in_tile * 64 + sw_col * 8 + elem;
}
__device__ __forceinline__ int hsm_off(int row, int col) {
    int stripe = col >> 6;
    int col_in_stripe = col & 63;
    return stripe * (BLOCK_B * 64) + sw128(row, col_in_stripe);
}
__device__ __forceinline__ int silu_off(int row, int col) { return sw128(row, col); }
__device__ __forceinline__ int weight_smem_off(int row, int col, int) {
    int stripe = col >> 6;
    int col_in_stripe = col & 63;
    return stripe * (R * 64) + sw128(row, col_in_stripe);
}

__device__ __forceinline__ uint32_t to_smem(const void* p) { return __cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t pk(bf16 lo, bf16 hi) {
    uint16_t l = *reinterpret_cast<uint16_t*>(&lo);
    uint16_t h = *reinterpret_cast<uint16_t*>(&hi);
    return (uint32_t(h) << 16) | uint32_t(l);
}
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = to_smem(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void
ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(to_smem(p)));
}
__device__ __forceinline__ void
ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(to_smem(p)));
}
__device__ __forceinline__ void
mma_m16n8k16(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
             uint32_t b0, uint32_t b1,
             float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
dlr_h100_prefill_b4096_v14_kernel(
    const bf16* __restrict__ X,
    bf16*       __restrict__ O,
    const bf16* __restrict__ Vs,
    const bf16* __restrict__ Us,
    int L
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int tg      = lane & 3;
    const int cta_id  = blockIdx.x;
    const int row0    = cta_id * BLOCK_B;

    extern __shared__ __align__(16) char smem_raw[];
    bf16*  h_smem      = reinterpret_cast<bf16*>(smem_raw);
    bf16*  V_smem      = h_smem + BLOCK_B * D;
    bf16*  U_smem      = V_smem + 2 * R * BLOCK_K_V;
    bf16*  silu_smem   = U_smem + 2 * R * BLOCK_N_U;
    float* rstd_smem   = reinterpret_cast<float*>(silu_smem + BLOCK_B * R);

    constexpr float INV_D = 1.0f / float(D);
    constexpr float EPS = 1e-6f;

    #pragma unroll
    for (int i = 0; i < BLOCK_B * D / (NTHREADS * 8); ++i) {
        int e = tid + i * NTHREADS;
        int b = e / (D / 8);
        int c = (e % (D / 8)) * 8;
        int gr = row0 + b;
        bf16* sptr = h_smem + hsm_off(b, c);
        const bf16* gptr = X + gr * D + c;
        cp_async_16(sptr, gptr);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    for (int l = 0; l < L; ++l) {
        const bf16* V_l = Vs + (size_t)l * R * D;
        const bf16* U_l = Us + (size_t)l * R * D;

        // RMSNorm
        constexpr int ROWS_PER_WARP = BLOCK_B / NWARPS;
        for (int rw = 0; rw < ROWS_PER_WARP; ++rw) {
            int b = warp_id * ROWS_PER_WARP + rw;
            float sq = 0.f;
            for (int d = lane; d < D; d += 32) {
                float v = __bfloat162float(h_smem[hsm_off(b, d)]);
                sq += v * v;
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(~0u, sq, o);
            if (lane == 0) rstd_smem[b] = rsqrtf(sq * INV_D + EPS);
        }
        __syncthreads();

        // V matmul (K-pair amortized)
        const int n_tile_v = warp_id;
        float z_acc[2][4];
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int i = 0; i < 4; ++i) z_acc[m][i] = 0.f;

        auto issue_v_chunk = [&](int kc, int buf) {
            bf16* V_dst = V_smem + buf * R * BLOCK_K_V;
            #pragma unroll
            for (int i = 0; i < R * BLOCK_K_V / (NTHREADS * 8); ++i) {
                int e = tid + i * NTHREADS;
                int r = e / (BLOCK_K_V / 8);
                int c = (e % (BLOCK_K_V / 8)) * 8;
                bf16* sptr = V_dst + weight_smem_off(r, c, BLOCK_K_V);
                const bf16* gptr = V_l + r * D + kc * BLOCK_K_V + c;
                cp_async_16(sptr, gptr);
            }
            cp_async_commit();
        };

        issue_v_chunk(0, 0);
        for (int kc = 0; kc < NUM_K_V; ++kc) {
            int buf_cur = kc & 1;
            int buf_nxt = 1 - buf_cur;
            bf16* V_cur = V_smem + buf_cur * R * BLOCK_K_V;
            if (kc + 1 < NUM_K_V) { issue_v_chunk(kc + 1, buf_nxt); cp_async_wait<1>(); }
            else { cp_async_wait<0>(); }
            __syncthreads();

            // K-pair loop: each iteration covers 2 K-tiles (32 K positions)
            #pragma unroll
            for (int kt = 0; kt < K_TILES_PER_VCHUNK; kt += 2) {
                int kb_chunk_0 = kt * K_TILE;
                int kb_global_0 = kc * BLOCK_K_V + kb_chunk_0;
                int kb_global_1 = kb_global_0 + K_TILE;

                // Load B for both K-tiles in one ldmatrix.x4:
                // b_k = kb_chunk_0 + (lane>>3)*8 covers K=[kb_chunk_0 .. +31]
                int b_n = n_tile_v * 8 + (lane & 7);
                int b_k = kb_chunk_0 + (lane >> 3) * 8;
                uint32_t b_r0, b_r1, b_r2, b_r3;
                ldmatrix_x4(b_r0, b_r1, b_r2, b_r3,
                            V_cur + weight_smem_off(b_n, b_k, BLOCK_K_V));
                // (b_r0, b_r1) → kt;  (b_r2, b_r3) → kt+1

                // mma for kt
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    int row_base = m * 16;
                    int arow = row_base + (lane & 15);
                    int acol = kb_global_0 + (lane >> 4) * 8;
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4(a0, a1, a2, a3, h_smem + hsm_off(arow, acol));
                    mma_m16n8k16(a0, a1, a2, a3, b_r0, b_r1,
                                 z_acc[m][0], z_acc[m][1], z_acc[m][2], z_acc[m][3]);
                }
                // mma for kt+1 (uses b_r2, b_r3)
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    int row_base = m * 16;
                    int arow = row_base + (lane & 15);
                    int acol = kb_global_1 + (lane >> 4) * 8;
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4(a0, a1, a2, a3, h_smem + hsm_off(arow, acol));
                    mma_m16n8k16(a0, a1, a2, a3, b_r2, b_r3,
                                 z_acc[m][0], z_acc[m][1], z_acc[m][2], z_acc[m][3]);
                }
            }
            __syncthreads();
        }

        // silu -> silu_smem
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int row_base = m * 16;
            int r0 = row_base + gid;
            int r1 = row_base + gid + 8;
            int c0 = n_tile_v * 8 + tg * 2;
            float rstd_r0 = rstd_smem[r0];
            float rstd_r1 = rstd_smem[r1];
            float zv0 = z_acc[m][0] * rstd_r0;
            float zv1 = z_acc[m][1] * rstd_r0;
            float zv2 = z_acc[m][2] * rstd_r1;
            float zv3 = z_acc[m][3] * rstd_r1;
            float s0 = zv0 / (1.0f + __expf(-zv0));
            float s1 = zv1 / (1.0f + __expf(-zv1));
            float s2 = zv2 / (1.0f + __expf(-zv2));
            float s3 = zv3 / (1.0f + __expf(-zv3));
            silu_smem[silu_off(r0, c0 + 0)] = __float2bfloat16(s0);
            silu_smem[silu_off(r0, c0 + 1)] = __float2bfloat16(s1);
            silu_smem[silu_off(r1, c0 + 0)] = __float2bfloat16(s2);
            silu_smem[silu_off(r1, c0 + 1)] = __float2bfloat16(s3);
        }
        __syncthreads();

        // Hoist silu into registers once per layer (constant across nc and n_sub).
        // Each warp needs A operand fragments at (m=0..1, kt=0..3) → 8 ldmatrix calls.
        uint32_t silu_reg[2][K_TILES_U][4];
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int row_base = m * 16;
            #pragma unroll
            for (int kt = 0; kt < K_TILES_U; ++kt) {
                int kb = kt * K_TILE;
                int arow = row_base + (lane & 15);
                int acol = kb + (lane >> 4) * 8;
                ldmatrix_x4(silu_reg[m][kt][0], silu_reg[m][kt][1],
                            silu_reg[m][kt][2], silu_reg[m][kt][3],
                            silu_smem + silu_off(arow, acol));
            }
        }

        // U matmul (chunk-local h_reg + K-pair amortized B)
        auto issue_u_chunk = [&](int nc, int buf) {
            int n_base = nc * BLOCK_N_U;
            bf16* U_dst = U_smem + buf * R * BLOCK_N_U;
            #pragma unroll
            for (int i = 0; i < R * BLOCK_N_U / (NTHREADS * 8); ++i) {
                int e = tid + i * NTHREADS;
                int r = e / (BLOCK_N_U / 8);
                int c = (e % (BLOCK_N_U / 8)) * 8;
                bf16* sptr = U_dst + weight_smem_off(r, c, BLOCK_N_U);
                const bf16* gptr = U_l + r * D + n_base + c;
                cp_async_16(sptr, gptr);
            }
            cp_async_commit();
        };

        issue_u_chunk(0, 0);

        for (int nc = 0; nc < NUM_N_U; ++nc) {
            int n_base = nc * BLOCK_N_U;
            int buf_cur = nc & 1;
            int buf_nxt = 1 - buf_cur;
            bf16* U_cur = U_smem + buf_cur * R * BLOCK_N_U;

            if (nc + 1 < NUM_N_U) { issue_u_chunk(nc + 1, buf_nxt); cp_async_wait<1>(); }
            else { cp_async_wait<0>(); }
            __syncthreads();

            uint32_t h_reg[2][N_PER_WARP_U][2];

            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                int row_base = m * 16;
                int r0 = row_base + gid;
                int r1 = row_base + gid + 8;
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_U + nw;
                    int c0 = n_base + n_sub * N_TILE + tg * 2;
                    h_reg[m][nw][0] = *reinterpret_cast<uint32_t*>(h_smem + hsm_off(r0, c0));
                    h_reg[m][nw][1] = *reinterpret_cast<uint32_t*>(h_smem + hsm_off(r1, c0));
                }
            }

            // dh_acc covers both n_sub (nw) so they accumulate in parallel:
            // 4 independent dep chains (m × nw), each of length 2 (kt+1 after kt).
            float dh_acc[2][N_PER_WARP_U][4];
            #pragma unroll
            for (int m = 0; m < 2; ++m)
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) dh_acc[m][nw][i] = 0.f;

            // K-pair loop: each iteration covers 2 K-tiles, both nw concurrently.
            #pragma unroll
            for (int kt = 0; kt < K_TILES_U; kt += 2) {
                int kb_0 = kt * K_TILE;

                // Load B for both K-tiles, for BOTH nw (independent SMEM addresses).
                int b_k = kb_0 + (lane >> 3) * 8 + (lane & 7);
                uint32_t b_r[N_PER_WARP_U][4];
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_U + nw;
                    int b_n = n_sub * N_TILE;
                    ldmatrix_x4_trans(b_r[nw][0], b_r[nw][1], b_r[nw][2], b_r[nw][3],
                                      U_cur + weight_smem_off(b_k, b_n, BLOCK_N_U));
                }

                // mma for kt: 2 m × N_PER_WARP_U nw = 4 independent mmas (parallel chains).
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    #pragma unroll
                    for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                        mma_m16n8k16(silu_reg[m][kt][0], silu_reg[m][kt][1],
                                     silu_reg[m][kt][2], silu_reg[m][kt][3],
                                     b_r[nw][0], b_r[nw][1],
                                     dh_acc[m][nw][0], dh_acc[m][nw][1],
                                     dh_acc[m][nw][2], dh_acc[m][nw][3]);
                    }
                }
                // mma for kt+1: same 4 chains continue.
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    #pragma unroll
                    for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                        mma_m16n8k16(silu_reg[m][kt+1][0], silu_reg[m][kt+1][1],
                                     silu_reg[m][kt+1][2], silu_reg[m][kt+1][3],
                                     b_r[nw][2], b_r[nw][3],
                                     dh_acc[m][nw][0], dh_acc[m][nw][1],
                                     dh_acc[m][nw][2], dh_acc[m][nw][3]);
                    }
                }
            }

            // Apply h_reg + dh_acc across both m and both nw.
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                    uint32_t cur_r0 = h_reg[m][nw][0];
                    uint32_t cur_r1 = h_reg[m][nw][1];
                    bf16* cur_r0_bf = reinterpret_cast<bf16*>(&cur_r0);
                    bf16* cur_r1_bf = reinterpret_cast<bf16*>(&cur_r1);
                    float h00 = __bfloat162float(cur_r0_bf[0]) + dh_acc[m][nw][0];
                    float h01 = __bfloat162float(cur_r0_bf[1]) + dh_acc[m][nw][1];
                    float h10 = __bfloat162float(cur_r1_bf[0]) + dh_acc[m][nw][2];
                    float h11 = __bfloat162float(cur_r1_bf[1]) + dh_acc[m][nw][3];
                    h_reg[m][nw][0] = pk(__float2bfloat16(h00), __float2bfloat16(h01));
                    h_reg[m][nw][1] = pk(__float2bfloat16(h10), __float2bfloat16(h11));
                }
            }

            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                int row_base = m * 16;
                int r0 = row_base + gid;
                int r1 = row_base + gid + 8;
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_U + nw;
                    int c0 = n_base + n_sub * N_TILE + tg * 2;
                    *reinterpret_cast<uint32_t*>(h_smem + hsm_off(r0, c0)) = h_reg[m][nw][0];
                    *reinterpret_cast<uint32_t*>(h_smem + hsm_off(r1, c0)) = h_reg[m][nw][1];
                }
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_B * D / (NTHREADS * 8); ++i) {
        int e = tid + i * NTHREADS;
        int b = e / (D / 8);
        int c = (e % (D / 8)) * 8;
        int gr = row0 + b;
        const bf16* sptr = h_smem + hsm_off(b, c);
        bf16* gptr = O + gr * D + c;
        *reinterpret_cast<int4*>(gptr) = *reinterpret_cast<const int4*>(sptr);
    }
}

torch::Tensor dlrnet_monolithic(torch::Tensor X, torch::Tensor Vs, torch::Tensor Us) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16);
    int B = X.size(0);
    TORCH_CHECK(B % BLOCK_B == 0);
    TORCH_CHECK(X.size(1) == D);
    int L = Vs.size(0);
    TORCH_CHECK(Vs.size(1) == R && Vs.size(2) == D);
    auto O = torch::empty_like(X);
    auto Xp = reinterpret_cast<const bf16*>(X.const_data_ptr<at::BFloat16>());
    auto Op = reinterpret_cast<bf16*>(O.data_ptr<at::BFloat16>());
    auto Vp = reinterpret_cast<const bf16*>(Vs.const_data_ptr<at::BFloat16>());
    auto Up = reinterpret_cast<const bf16*>(Us.const_data_ptr<at::BFloat16>());
    int smem_bytes = BLOCK_B * D * 2
                   + 2 * R * BLOCK_K_V * 2
                   + 2 * R * BLOCK_N_U * 2
                   + BLOCK_B * R * 2
                   + BLOCK_B * 4
                   + 256;
    cudaFuncSetAttribute(dlr_h100_prefill_b4096_v14_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    int grid = B / BLOCK_B;
    dlr_h100_prefill_b4096_v14_kernel<<<grid, NTHREADS, smem_bytes,
                                      at::cuda::getCurrentCUDAStream()>>>(Xp, Op, Vp, Up, L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlrnet_monolithic", &dlrnet_monolithic,
          "1.7B DLR-Net monolithic fused prefill");
}
