//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Cooperative fused DLR-Net prefill kernel (D=5120, R=256).
//
// Same cooperative design as cooperative.cu, scaled for D=5120, R=256. Runs one
// forward pass of a stack of L DLR-Net residual layers, each computing
//   h += U @ silu((h @ V) * rstd),   rstd = 1/sqrt(mean(h^2) + eps),
// with the RMSNorm gain folded into V and the residual scale folded into U. A
// group of NP CTAs cooperate on one tile of BLOCK_B rows; each CTA owns one 1/NP
// stripe of D and does the V matmul over its own K-stripe and the U matmul over
// its own N-stripe (the stripes coincide, so no cross-CTA exchange of h is
// needed). A per-layer atomic-spin barrier plus a global reduction buffer combine
// the partial V projection (ZBUF) and RMSNorm sum-of-squares (SQBUF) across CTAs,
// and V/U weights for the next layer are prefetched across the layer boundary.
// As in cooperative_8b.cu, h_smem holds only each CTA's own stripe; the cross-CTA
// Z reduction streams the NP partials from the global buffer directly into
// registers instead of staging them in shared memory.
//
// Compile-time config: D=CFG_D, R=CFG_R, BLOCK_B=CFG_BLOCK_B,
// BLOCK_K_V=CFG_BLOCK_K_V, BLOCK_N_U=CFG_BLOCK_N_U, NP=CFG_NP, NWARPS=CFG_NWARPS,
// FENCE=CFG_FENCE, CTAS_PER_SM=CFG_CTAS_PER_SM, U_BUFS=CFG_U_BUFS.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;

#ifndef CFG_D
#define CFG_D 5120
#endif
#ifndef CFG_R
#define CFG_R 256
#endif
#ifndef CFG_BLOCK_B
#define CFG_BLOCK_B 16
#endif
#ifndef CFG_BLOCK_K_V
#define CFG_BLOCK_K_V 64
#endif
#ifndef CFG_BLOCK_N_U
#define CFG_BLOCK_N_U 64
#endif
#ifndef CFG_NP
#define CFG_NP 20
#endif
#ifndef CFG_FENCE
#define CFG_FENCE 0
#endif
#ifndef CFG_CTAS_PER_SM
#define CFG_CTAS_PER_SM 1
#endif
// U single-buffer mode: shrinks SMEM by R*BLOCK_N_U*2 bytes (32 KB at R=256, BN=64)
// to fit 2 CTAs per SM, doubling warps/scheduler. Trade-off: U cp.async stalls
// inline per chunk instead of overlapping with previous chunk's compute.
#ifndef CFG_U_BUFS
#define CFG_U_BUFS 2
#endif
#ifndef CFG_NWARPS
#define CFG_NWARPS 8
#endif

constexpr int D = CFG_D;
constexpr int R = CFG_R;
constexpr int BLOCK_B = CFG_BLOCK_B;
constexpr int M_UNROLL = BLOCK_B / 16;
constexpr int NWARPS = CFG_NWARPS;
constexpr int NTHREADS = NWARPS * 32;
constexpr int BLOCK_K_V = CFG_BLOCK_K_V;
constexpr int BLOCK_N_U = CFG_BLOCK_N_U;
constexpr int NUM_K_V = D / BLOCK_K_V;
constexpr int NUM_N_U = D / BLOCK_N_U;
constexpr int NP = CFG_NP;
constexpr int K_CHUNKS_PER_CTA = NUM_K_V / NP;
constexpr int N_CHUNKS_PER_CTA = NUM_N_U / NP;
constexpr int K_TILE = 16;
constexpr int N_TILE = 8;
constexpr int K_TILES_PER_VCHUNK = BLOCK_K_V / K_TILE;
constexpr int K_TILES_U = R / K_TILE;
constexpr int N_TILES_PER_CHUNK = BLOCK_N_U / N_TILE;
constexpr int N_PER_WARP_U = N_TILES_PER_CHUNK / NWARPS;
constexpr int N_TILES_V = R / N_TILE;
constexpr int N_PER_WARP_V = N_TILES_V / NWARPS;  // R=256: 4 col tiles per warp

// Per-CTA local h-stripe width: own K-stripe == own N-stripe (same d range).
constexpr int H_STRIPE_W = D / NP;

static_assert(BLOCK_B % 16 == 0, "");
static_assert(BLOCK_B >= NWARPS, "");
static_assert(D % BLOCK_K_V == 0, "");
static_assert(D % BLOCK_N_U == 0, "");
static_assert(NUM_N_U % NP == 0, "NUM_N_U must be divisible by NP");
static_assert(NUM_K_V % NP == 0, "NUM_K_V must be divisible by NP");
static_assert(K_CHUNKS_PER_CTA >= 1, "");
static_assert(N_CHUNKS_PER_CTA >= 1, "");
static_assert(R * BLOCK_K_V % (NTHREADS * 8) == 0, "");
static_assert(R * BLOCK_N_U % (NTHREADS * 8) == 0, "");
static_assert(H_STRIPE_W % 64 == 0, "h_smem layout uses 64-wide swizzled stripes");
static_assert(D % NP == 0, "D must be divisible by NP for own-stripe layout");
// K-stripe and N-stripe widths must match (same d range owned by each CTA).
static_assert(K_CHUNKS_PER_CTA * BLOCK_K_V == N_CHUNKS_PER_CTA * BLOCK_N_U,
              "K-stripe must equal N-stripe");
static_assert(K_CHUNKS_PER_CTA * BLOCK_K_V == H_STRIPE_W, "");
static_assert(N_PER_WARP_V >= 1, "R / N_TILE / NWARPS must be >= 1");

__device__ __forceinline__ int sw128(int row, int col) {
    int tile_idx     = row >> 3;
    int row_in_tile  = row & 7;
    int core_col     = col >> 3;
    int sw_col       = core_col ^ row_in_tile;
    int elem         = col & 7;
    return tile_idx * 512 + row_in_tile * 64 + sw_col * 8 + elem;
}
// hsm_off uses LOCAL col within own stripe ([0, H_STRIPE_W)).
__device__ __forceinline__ int hsm_off_local(int row, int col_local) {
    int stripe = col_local >> 6;
    int col_in_stripe = col_local & 63;
    return stripe * (BLOCK_B * 64) + sw128(row, col_in_stripe);
}
__device__ __forceinline__ int silu_off(int row, int col) {
    int stripe = col >> 6;
    int col_in_stripe = col & 63;
    return stripe * (BLOCK_B * 64) + sw128(row, col_in_stripe);
}
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

extern "C" __global__ void __launch_bounds__(NTHREADS, CFG_CTAS_PER_SM)
dlr_h100_prefill_b_mp_v8_32b_kernel(
    const bf16* __restrict__ X,
    bf16*       __restrict__ O,
    float*      __restrict__ ZBUF,        // ((B/BLOCK_B), L, NP, BLOCK_B, R) fp32
    float*      __restrict__ SQBUF,       // ((B/BLOCK_B), L, NP, BLOCK_B) fp32
    int*        __restrict__ BAR_V,       // ((B/BLOCK_B) * L) ints, zero-init
    const bf16* __restrict__ Vs,
    const bf16* __restrict__ Us,
    int L
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int tg      = lane & 3;
    const int cta_lin = blockIdx.x;
    const int cta_id_n = cta_lin % NP;
    const int cta_id_b = cta_lin / NP;
    const int row0    = cta_id_b * BLOCK_B;
    const int kc_start = cta_id_n * K_CHUNKS_PER_CTA;
    const int kc_end   = kc_start + K_CHUNKS_PER_CTA;
    const int n_chunk_start = cta_id_n * N_CHUNKS_PER_CTA;
    const int n_chunk_end   = n_chunk_start + N_CHUNKS_PER_CTA;
    const int own_d_start   = cta_id_n * H_STRIPE_W;           // global d-index of stripe[0]

    extern __shared__ __align__(16) char smem_raw[];
    bf16*  h_smem      = reinterpret_cast<bf16*>(smem_raw);                    // [BB, H_STRIPE_W]
    bf16*  V_smem      = h_smem + BLOCK_B * H_STRIPE_W;                        // [2, R, BK]
    bf16*  U_smem      = V_smem + 2 * R * BLOCK_K_V;                           // [CFG_U_BUFS, R, BN]
    bf16*  silu_smem   = U_smem + CFG_U_BUFS * R * BLOCK_N_U;                  // [BB, R]
    float* rstd_smem   = reinterpret_cast<float*>(silu_smem + BLOCK_B * R);    // [BB]

    constexpr float INV_D = 1.0f / float(D);
    constexpr float EPS = 1e-6f;

    // Initial X load: only OWN K-stripe (cols [own_d_start, own_d_start + H_STRIPE_W))
    {
        constexpr int H_BYTES = BLOCK_B * H_STRIPE_W * 2;
        constexpr int H_VEC   = H_BYTES / 16;
        #pragma unroll
        for (int i = 0; i < H_VEC / NTHREADS; ++i) {
            int e = tid + i * NTHREADS;
            int b = e / (H_STRIPE_W / 8);
            int c_local = (e % (H_STRIPE_W / 8)) * 8;
            int gr = row0 + b;
            bf16* sptr = h_smem + hsm_off_local(b, c_local);
            const bf16* gptr = X + gr * D + own_d_start + c_local;
            cp_async_16(sptr, gptr);
        }
        cp_async_commit();
        cp_async_wait<0>();
    }
    __syncthreads();

    // Pre-layer-0 prefetch: V chunk 0 of layer 0 → V_smem buf 0.
    {
        const bf16* V_l0 = Vs + (size_t)0 * R * D;
        bf16* V_dst = V_smem + 0 * R * BLOCK_K_V;
        #pragma unroll
        for (int i = 0; i < R * BLOCK_K_V / (NTHREADS * 8); ++i) {
            int e = tid + i * NTHREADS;
            int r = e / (BLOCK_K_V / 8);
            int c = (e % (BLOCK_K_V / 8)) * 8;
            bf16* sptr = V_dst + weight_smem_off(r, c, BLOCK_K_V);
            const bf16* gptr = V_l0 + r * D + kc_start * BLOCK_K_V + c;
            cp_async_16(sptr, gptr);
        }
        cp_async_commit();
    }

    // Pre-layer-0 prefetch: U chunk 0 of layer 0 → U_smem buf 0.
    {
        const bf16* U_l0 = Us + (size_t)0 * R * D;
        bf16* U_dst = U_smem + 0 * R * BLOCK_N_U;
        #pragma unroll
        for (int i = 0; i < R * BLOCK_N_U / (NTHREADS * 8); ++i) {
            int e = tid + i * NTHREADS;
            int r = e / (BLOCK_N_U / 8);
            int c = (e % (BLOCK_N_U / 8)) * 8;
            bf16* sptr = U_dst + weight_smem_off(r, c, BLOCK_N_U);
            const bf16* gptr = U_l0 + r * D + n_chunk_start * BLOCK_N_U + c;
            cp_async_16(sptr, gptr);
        }
        cp_async_commit();
    }

    for (int l = 0; l < L; ++l) {
        const bf16* V_l = Vs + (size_t)l * R * D;
        const bf16* U_l = Us + (size_t)l * R * D;

        // ---- Partial sum-of-squares over OWN K-stripe of h ----
        // h_smem holds local cols 0..H_STRIPE_W (already own K-stripe).
        constexpr int ROWS_PER_WARP = BLOCK_B / NWARPS;
        for (int rw = 0; rw < ROWS_PER_WARP; ++rw) {
            int b = warp_id * ROWS_PER_WARP + rw;
            float sq = 0.f;
            for (int d_local = lane; d_local < H_STRIPE_W; d_local += 32) {
                float v = __bfloat162float(h_smem[hsm_off_local(b, d_local)]);
                sq += v * v;
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(~0u, sq, o);
            if (lane == 0) rstd_smem[b] = sq;
        }
        __syncthreads();

        // ---- V matmul (own K-stripe only: kc ∈ [kc_start, kc_end)) ----
        // 32B: each warp owns N_PER_WARP_V column tiles of z (R / N_TILE / NWARPS).
        float z_acc[M_UNROLL][N_PER_WARP_V][4];
        #pragma unroll
        for (int m = 0; m < M_UNROLL; ++m)
            #pragma unroll
            for (int nw = 0; nw < N_PER_WARP_V; ++nw)
                #pragma unroll
                for (int i = 0; i < 4; ++i) z_acc[m][nw][i] = 0.f;

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

        for (int kc = kc_start; kc < kc_end; ++kc) {
            int local_idx = kc - kc_start;
            int buf_cur = local_idx & 1;
            int buf_nxt = 1 - buf_cur;
            bf16* V_cur = V_smem + buf_cur * R * BLOCK_K_V;
            if (kc + 1 < kc_end) { issue_v_chunk(kc + 1, buf_nxt); cp_async_wait<1>(); }
            else { cp_async_wait<0>(); }
            __syncthreads();

            #pragma unroll
            for (int kt = 0; kt < K_TILES_PER_VCHUNK; kt += 2) {
                int kb_chunk_0 = kt * K_TILE;
                int kb_global_0 = kc * BLOCK_K_V + kb_chunk_0;
                int kb_global_1 = kb_global_0 + K_TILE;
                int kb_local_0 = kb_global_0 - own_d_start;
                int kb_local_1 = kb_global_1 - own_d_start;

                // Load B for both K-tiles, all N_PER_WARP_V n_sub.
                uint32_t b_r[N_PER_WARP_V][4];
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_V; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_V + nw;
                    int b_n = n_sub * 8 + (lane & 7);
                    int b_k = kb_chunk_0 + (lane >> 3) * 8;
                    ldmatrix_x4(b_r[nw][0], b_r[nw][1], b_r[nw][2], b_r[nw][3],
                                V_cur + weight_smem_off(b_n, b_k, BLOCK_K_V));
                }

                #pragma unroll
                for (int m = 0; m < M_UNROLL; ++m) {
                    int row_base = m * 16;
                    int arow = row_base + (lane & 15);
                    int acol = kb_local_0 + (lane >> 4) * 8;
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4(a0, a1, a2, a3, h_smem + hsm_off_local(arow, acol));
                    #pragma unroll
                    for (int nw = 0; nw < N_PER_WARP_V; ++nw) {
                        mma_m16n8k16(a0, a1, a2, a3, b_r[nw][0], b_r[nw][1],
                                     z_acc[m][nw][0], z_acc[m][nw][1],
                                     z_acc[m][nw][2], z_acc[m][nw][3]);
                    }
                }
                #pragma unroll
                for (int m = 0; m < M_UNROLL; ++m) {
                    int row_base = m * 16;
                    int arow = row_base + (lane & 15);
                    int acol = kb_local_1 + (lane >> 4) * 8;
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4(a0, a1, a2, a3, h_smem + hsm_off_local(arow, acol));
                    #pragma unroll
                    for (int nw = 0; nw < N_PER_WARP_V; ++nw) {
                        mma_m16n8k16(a0, a1, a2, a3, b_r[nw][2], b_r[nw][3],
                                     z_acc[m][nw][0], z_acc[m][nw][1],
                                     z_acc[m][nw][2], z_acc[m][nw][3]);
                    }
                }
            }
            __syncthreads();
        }

        // ---- V-reduce + sq-reduce: write partials to ZBUF & SQBUF, single barrier ----
        {
            float* zb_own = ZBUF + ((cta_id_b * L + l) * NP + cta_id_n) * (BLOCK_B * R);
            #pragma unroll
            for (int m = 0; m < M_UNROLL; ++m) {
                int row_base = m * 16;
                int r0 = row_base + gid;
                int r1 = row_base + gid + 8;
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_V; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_V + nw;
                    int c0 = n_sub * 8 + tg * 2;
                    // Vec store: (s0,s1) and (s2,s3) are int2-aligned (c0 % 2 == 0).
                    float2 p0 = {z_acc[m][nw][0], z_acc[m][nw][1]};
                    float2 p1 = {z_acc[m][nw][2], z_acc[m][nw][3]};
                    *reinterpret_cast<float2*>(&zb_own[r0 * R + c0]) = p0;
                    *reinterpret_cast<float2*>(&zb_own[r1 * R + c0]) = p1;
                }
            }
            float* sb_own = SQBUF + ((cta_id_b * L + l) * NP + cta_id_n) * BLOCK_B;
            if (tid < BLOCK_B) sb_own[tid] = rstd_smem[tid];
        }
        __syncthreads();
        #if CFG_FENCE
        __threadfence();
        #endif
        if (tid == 0) atomicAdd(&BAR_V[cta_id_b * L + l], 1);
        if (tid == 0) {
            volatile int* p = &BAR_V[cta_id_b * L + l];
            while (*p < NP) {}
        }
        __syncthreads();

        // ---- Streaming Z reduce: accumulate NP partials into z_acc registers.
        // L2 caches the per-layer ZBUF slice.
        {
            float* zb_block_base = ZBUF + (cta_id_b * L + l) * NP * (BLOCK_B * R);
            #pragma unroll
            for (int m = 0; m < M_UNROLL; ++m) {
                int row_base = m * 16;
                int r0 = row_base + gid;
                int r1 = row_base + gid + 8;
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_V; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_V + nw;
                    int c0 = n_sub * 8 + tg * 2;
                    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
                    #pragma unroll
                    for (int n = 0; n < NP; ++n) {
                        float* zb_n = zb_block_base + n * (BLOCK_B * R);
                        float2 p0 = *reinterpret_cast<float2*>(&zb_n[r0 * R + c0]);
                        float2 p1 = *reinterpret_cast<float2*>(&zb_n[r1 * R + c0]);
                        s0 += p0.x; s1 += p0.y;
                        s2 += p1.x; s3 += p1.y;
                    }
                    z_acc[m][nw][0] = s0;
                    z_acc[m][nw][1] = s1;
                    z_acc[m][nw][2] = s2;
                    z_acc[m][nw][3] = s3;
                }
            }
        }

        // Sum SQ partials → final rstd in rstd_smem.
        if (tid < BLOCK_B) {
            float* sq_base = SQBUF + (cta_id_b * L + l) * NP * BLOCK_B;
            float total = 0.f;
            #pragma unroll
            for (int n = 0; n < NP; ++n) total += sq_base[n * BLOCK_B + tid];
            rstd_smem[tid] = rsqrtf(total * INV_D + EPS);
        }
        __syncthreads();

        // ---- silu (replicated full) ----
        #pragma unroll
        for (int m = 0; m < M_UNROLL; ++m) {
            int row_base = m * 16;
            int r0 = row_base + gid;
            int r1 = row_base + gid + 8;
            float rstd_r0 = rstd_smem[r0];
            float rstd_r1 = rstd_smem[r1];
            #pragma unroll
            for (int nw = 0; nw < N_PER_WARP_V; ++nw) {
                int n_sub = warp_id * N_PER_WARP_V + nw;
                int c0 = n_sub * 8 + tg * 2;
                float zv0 = z_acc[m][nw][0] * rstd_r0;
                float zv1 = z_acc[m][nw][1] * rstd_r0;
                float zv2 = z_acc[m][nw][2] * rstd_r1;
                float zv3 = z_acc[m][nw][3] * rstd_r1;
                float s0 = zv0 / (1.0f + __expf(-zv0));
                float s1 = zv1 / (1.0f + __expf(-zv1));
                float s2 = zv2 / (1.0f + __expf(-zv2));
                float s3 = zv3 / (1.0f + __expf(-zv3));
                silu_smem[silu_off(r0, c0 + 0)] = __float2bfloat16(s0);
                silu_smem[silu_off(r0, c0 + 1)] = __float2bfloat16(s1);
                silu_smem[silu_off(r1, c0 + 0)] = __float2bfloat16(s2);
                silu_smem[silu_off(r1, c0 + 1)] = __float2bfloat16(s3);
            }
        }
        __syncthreads();

        uint32_t silu_reg[M_UNROLL][K_TILES_U][4];
        #pragma unroll
        for (int m = 0; m < M_UNROLL; ++m) {
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

        // ---- U matmul (own N-stripe only) ----
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

        for (int nc = n_chunk_start; nc < n_chunk_end; ++nc) {
            int n_base = nc * BLOCK_N_U;                     // global d-index
            int n_base_local = n_base - own_d_start;          // local d in h_smem
            int local_idx = nc - n_chunk_start;
#if CFG_U_BUFS == 2
            int buf_cur = local_idx & 1;
            int buf_nxt = 1 - buf_cur;
            bf16* U_cur = U_smem + buf_cur * R * BLOCK_N_U;
            if (nc + 1 < n_chunk_end) { issue_u_chunk(nc + 1, buf_nxt); cp_async_wait<1>(); }
            else { cp_async_wait<0>(); }
#else
            // Single-buf U: just wait for current chunk's prefetch. Next chunk
            // issues at end of this iteration (after we're done reading U_cur).
            bf16* U_cur = U_smem;
            cp_async_wait<0>();
#endif
            __syncthreads();

            uint32_t h_reg[M_UNROLL][N_PER_WARP_U][2];
            #pragma unroll
            for (int m = 0; m < M_UNROLL; ++m) {
                int row_base = m * 16;
                int r0 = row_base + gid;
                int r1 = row_base + gid + 8;
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_U + nw;
                    int c0_local = n_base_local + n_sub * N_TILE + tg * 2;
                    h_reg[m][nw][0] = *reinterpret_cast<uint32_t*>(h_smem + hsm_off_local(r0, c0_local));
                    h_reg[m][nw][1] = *reinterpret_cast<uint32_t*>(h_smem + hsm_off_local(r1, c0_local));
                }
            }

            float dh_acc[M_UNROLL][N_PER_WARP_U][4];
            #pragma unroll
            for (int m = 0; m < M_UNROLL; ++m)
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) dh_acc[m][nw][i] = 0.f;

            #pragma unroll
            for (int kt = 0; kt < K_TILES_U; kt += 2) {
                int kb_0 = kt * K_TILE;
                int b_k = kb_0 + (lane >> 3) * 8 + (lane & 7);
                uint32_t b_r[N_PER_WARP_U][4];
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_U + nw;
                    int b_n = n_sub * N_TILE;
                    ldmatrix_x4_trans(b_r[nw][0], b_r[nw][1], b_r[nw][2], b_r[nw][3],
                                      U_cur + weight_smem_off(b_k, b_n, BLOCK_N_U));
                }
                #pragma unroll
                for (int m = 0; m < M_UNROLL; ++m) {
                    #pragma unroll
                    for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                        mma_m16n8k16(silu_reg[m][kt][0], silu_reg[m][kt][1],
                                     silu_reg[m][kt][2], silu_reg[m][kt][3],
                                     b_r[nw][0], b_r[nw][1],
                                     dh_acc[m][nw][0], dh_acc[m][nw][1],
                                     dh_acc[m][nw][2], dh_acc[m][nw][3]);
                    }
                }
                #pragma unroll
                for (int m = 0; m < M_UNROLL; ++m) {
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

            #pragma unroll
            for (int m = 0; m < M_UNROLL; ++m) {
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
            for (int m = 0; m < M_UNROLL; ++m) {
                int row_base = m * 16;
                int r0 = row_base + gid;
                int r1 = row_base + gid + 8;
                #pragma unroll
                for (int nw = 0; nw < N_PER_WARP_U; ++nw) {
                    int n_sub = warp_id * N_PER_WARP_U + nw;
                    int c0_local = n_base_local + n_sub * N_TILE + tg * 2;
                    *reinterpret_cast<uint32_t*>(h_smem + hsm_off_local(r0, c0_local)) = h_reg[m][nw][0];
                    *reinterpret_cast<uint32_t*>(h_smem + hsm_off_local(r1, c0_local)) = h_reg[m][nw][1];
                }
            }
            __syncthreads();
#if CFG_U_BUFS == 1
            // Single-buf: issue next chunk now that compute on this chunk is done.
            if (nc + 1 < n_chunk_end) { issue_u_chunk(nc + 1, 0); }
#endif
        }

        // Cross-layer V chunk 0 + U chunk 0 prefetch
        if (l + 1 < L) {
            const bf16* V_l_next = Vs + (size_t)(l + 1) * R * D;
            bf16* V_dst = V_smem + 0 * R * BLOCK_K_V;
            #pragma unroll
            for (int i = 0; i < R * BLOCK_K_V / (NTHREADS * 8); ++i) {
                int e = tid + i * NTHREADS;
                int r = e / (BLOCK_K_V / 8);
                int c = (e % (BLOCK_K_V / 8)) * 8;
                bf16* sptr = V_dst + weight_smem_off(r, c, BLOCK_K_V);
                const bf16* gptr = V_l_next + r * D + kc_start * BLOCK_K_V + c;
                cp_async_16(sptr, gptr);
            }
            cp_async_commit();

            const bf16* U_l_next = Us + (size_t)(l + 1) * R * D;
            bf16* U_dst = U_smem + 0 * R * BLOCK_N_U;
            #pragma unroll
            for (int i = 0; i < R * BLOCK_N_U / (NTHREADS * 8); ++i) {
                int e = tid + i * NTHREADS;
                int r = e / (BLOCK_N_U / 8);
                int c = (e % (BLOCK_N_U / 8)) * 8;
                bf16* sptr = U_dst + weight_smem_off(r, c, BLOCK_N_U);
                const bf16* gptr = U_l_next + r * D + n_chunk_start * BLOCK_N_U + c;
                cp_async_16(sptr, gptr);
            }
            cp_async_commit();
        }
    }

    // Final write: each CTA writes its own n-stripe (= own d-stripe) to O.
    {
        constexpr int total_bytes = BLOCK_B * H_STRIPE_W * 2;
        constexpr int total_vec   = total_bytes / 16;
        #pragma unroll
        for (int i = tid; i < total_vec; i += NTHREADS) {
            int b = i / (H_STRIPE_W / 8);
            int c_local = (i % (H_STRIPE_W / 8)) * 8;
            int gr = row0 + b;
            int gc = own_d_start + c_local;
            int4 v = *reinterpret_cast<const int4*>(h_smem + hsm_off_local(b, c_local));
            *reinterpret_cast<int4*>(O + (size_t)gr * D + gc) = v;
        }
    }
}

torch::Tensor dlrnet_cooperative_32b(torch::Tensor X, torch::Tensor Vs, torch::Tensor Us) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16);
    int B = X.size(0);
    TORCH_CHECK(B % BLOCK_B == 0, "B must be divisible by BLOCK_B=", BLOCK_B);
    TORCH_CHECK(X.size(1) == D);
    int L = Vs.size(0);
    TORCH_CHECK(Vs.size(1) == R && Vs.size(2) == D);
    auto O = torch::empty_like(X);
    auto opts_int  = X.options().dtype(torch::kInt32);
    auto opts_f32  = X.options().dtype(torch::kFloat32);
    int n_b_tiles = B / BLOCK_B;
    auto BAR_V = torch::zeros({n_b_tiles * L}, opts_int);
    auto ZBUF  = torch::empty({(int64_t)n_b_tiles * L * NP * BLOCK_B * R}, opts_f32);
    auto SQBUF = torch::empty({(int64_t)n_b_tiles * L * NP * BLOCK_B}, opts_f32);

    auto Xp  = reinterpret_cast<const bf16*>(X.const_data_ptr<at::BFloat16>());
    auto Op  = reinterpret_cast<bf16*>(O.data_ptr<at::BFloat16>());
    auto BVp = BAR_V.data_ptr<int>();
    auto Zp  = ZBUF.data_ptr<float>();
    auto SQp = SQBUF.data_ptr<float>();
    auto Vp  = reinterpret_cast<const bf16*>(Vs.const_data_ptr<at::BFloat16>());
    auto Up  = reinterpret_cast<const bf16*>(Us.const_data_ptr<at::BFloat16>());

    int smem_bytes = BLOCK_B * H_STRIPE_W * 2     // h_smem (own stripe only)
                   + 2 * R * BLOCK_K_V * 2         // V_smem 2-buf
                   + CFG_U_BUFS * R * BLOCK_N_U * 2 // U_smem (1 or 2 buf)
                   + BLOCK_B * R * 2               // silu_smem
                   + BLOCK_B * 4                   // rstd_smem
                   + 256;
    cudaError_t err = cudaFuncSetAttribute(dlr_h100_prefill_b_mp_v8_32b_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err),
                " (smem_bytes=", smem_bytes, ")");
    int grid = n_b_tiles * NP;
    dlr_h100_prefill_b_mp_v8_32b_kernel<<<grid, NTHREADS, smem_bytes,
                                          at::cuda::getCurrentCUDAStream()>>>(
        Xp, Op, Zp, SQp, BVp, Vp, Up, L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlrnet_cooperative_32b", &dlrnet_cooperative_32b,
          "Cooperative fused DLR-Net prefill (D=5120, R=256)");
    m.attr("BLOCK_B") = BLOCK_B;
    m.attr("NP") = NP;
    m.attr("H_STRIPE_W") = H_STRIPE_W;
}
