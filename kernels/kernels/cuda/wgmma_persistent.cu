//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Fused DLR-Net prefill kernel with WGMMA + persistent balanced-wave grid
// (D=2048, R=64), for large batch.
//
// Runs one forward pass of an L-layer DLR-Net block stack. Each layer computes
// h += alpha * U @ silu(V @ rmsnorm(h)); with the RMS gain folded into V and
// alpha folded into U this becomes z = (h @ V) * rstd, then h += silu(z) @ U.
// The V and U matmuls use WGMMA tensor-core instructions with double-buffered
// TMA weight/activation loads and warp-group specialization (one warpgroup
// drives the matmuls while all warps fold the sum-of-squares for rmsnorm).
// The launch uses a persistent grid: the number of CTAs is derived from an
// occupancy query so that the row-tiles split into the minimum number of
// equally-full hardware waves, keeping memory bandwidth saturated; each CTA
// strides over multiple row-tiles.
//
// Compile-time config: D=CFG_D, R=CFG_R, rows/tile=CFG_BB. Requires B % BB == 0.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

#ifndef CFG_D
#define CFG_D 2048
#endif
#ifndef CFG_R
#define CFG_R 64
#endif
#ifndef CFG_BB
#define CFG_BB 64
#endif

constexpr int D = CFG_D;
constexpr int R = CFG_R;
constexpr int BB = CFG_BB;
constexpr int NWARPS = 8;   // 2 warpgroups: WG0 = compute, WG1 = sq/rstd
constexpr int NTHREADS = NWARPS * 32;
constexpr int STRIPE_W = 64;
constexpr int N_STRIPES_D = D / STRIPE_W;
constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = 64;
constexpr int WGMMA_K = 16;
constexpr int K_TILES_STRIPE = STRIPE_W / WGMMA_K;

static_assert(BB == WGMMA_M, "");
static_assert(R == WGMMA_N, "");

__device__ __forceinline__ int sw128(int row, int col) {
    int tile_idx = row >> 3, row_in_tile = row & 7;
    int core_col = col >> 3, sw_col = core_col ^ row_in_tile;
    int elem = col & 7;
    return tile_idx * 512 + row_in_tile * 64 + sw_col * 8 + elem;
}
__device__ __forceinline__ int hs_off(int row, int col) { return sw128(row, col); }
__device__ __forceinline__ int silu_off(int row, int col) { return sw128(row, col); }

__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(to_smem(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(to_smem(mbar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t parity) {
    asm volatile(
        "{\n .reg .pred p;\n"
        "L_wait_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra L_wait_%=;\n"
        "}"
        :: "r"(to_smem(mbar)), "r"(parity)
        : "memory");
}
__device__ __forceinline__ void tma_load_2d(
    void* smem_dst, const CUtensorMap* desc,
    int32_t coord_col, int32_t coord_row, uint64_t* mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"(to_smem(smem_dst)),
           "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(to_smem(mbar)),
           "r"(coord_col), "r"(coord_row)
        : "memory");
}
__device__ __forceinline__ void tma_store_2d(
    const void* smem_src, const CUtensorMap* desc,
    int32_t coord_col, int32_t coord_row
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(to_smem(smem_src)),
           "r"(coord_col), "r"(coord_row)
        : "memory");
}
__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;\n" ::);
}
template<int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N));
}

union GmmaDescriptor {
    uint64_t desc;
    struct {
        uint16_t start_address_       : 14, : 2;
        uint16_t leading_byte_offset_ : 14, : 2;
        uint16_t stride_byte_offset_  : 14, : 2;
        uint8_t  : 1, base_offset_ : 3, : 4;
        uint8_t  : 6, layout_type_ : 2;
    } bf;
};
__device__ __forceinline__ uint64_t
make_desc_sw128b(void* smem_ptr, uint32_t sbo_u128, uint32_t lbo_u128) {
    GmmaDescriptor d; d.desc = 0;
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    d.bf.start_address_       = (uint16_t)((addr & 0x3FFFF) >> 4);
    d.bf.leading_byte_offset_ = (uint16_t)lbo_u128;
    d.bf.stride_byte_offset_  = (uint16_t)sbo_u128;
    d.bf.layout_type_         = 1;
    d.bf.base_offset_         = 0;
    return d.desc;
}
__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template<int N>
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void
wgmma_m64n64k16(uint64_t desc_a, uint64_t desc_b, float* d, int scale_d, int trans_b) {
    if (trans_b == 0) {
        asm volatile(
            "{\n .reg .pred p; setp.ne.b32 p, %34, 0;\n"
            " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16"
            " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "  %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            " %32, %33, p, 1, 1, 0, 0;\n}\n"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
              "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
              "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
              "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
            : "l"(desc_a), "l"(desc_b), "r"(scale_d));
    } else {
        asm volatile(
            "{\n .reg .pred p; setp.ne.b32 p, %34, 0;\n"
            " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16"
            " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "  %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            " %32, %33, p, 1, 1, 0, 1;\n}\n"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
              "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
              "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
              "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
            : "l"(desc_a), "l"(desc_b), "r"(scale_d));
    }
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
dlr_h100_prefill_17b_v30_balanced_kernel(
    const __grid_constant__ CUtensorMap V_desc,
    const __grid_constant__ CUtensorMap U_desc,
    const __grid_constant__ CUtensorMap O_load_desc,
    const __grid_constant__ CUtensorMap O_store_desc,
    const bf16* __restrict__ X,
    bf16*       __restrict__ O,
    int L,
    int num_blocks)   // = B / BB
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int tg = lane & 3;
    const int wg_id = warp_id >> 2;    // 0 or 1
    const int wg_warp = warp_id & 3;   // 0..3 within warpgroup
    const int cta = blockIdx.x;
    const int grid = gridDim.x;
    // Persistent outer loop — each CTA processes multiple row-blocks.
    // Stride-mapped: CTA g handles blocks g, g+grid, g+2*grid, ... < num_blocks.
    // Stride mapping (vs contiguous chunks) also gives better L2 locality
    // between adjacent CTA-blocks: consecutive-in-time blocks are far apart
    // in memory, reducing conflict misses.

    extern __shared__ __align__(16) char smem_raw[];
    // Scalar pairs (not arrays) so ternary select `cur ? _1 : _0` compiles
    // to selp, avoiding local-memory spill of pointer arrays.
    bf16* const h_smem_0 = reinterpret_cast<bf16*>(smem_raw);
    bf16* const h_smem_1 = h_smem_0 + BB * STRIPE_W;
    bf16* const W_smem_0 = h_smem_1 + BB * STRIPE_W;
    bf16* const W_smem_1 = W_smem_0 + R * STRIPE_W;
    bf16*  silu_smem  = W_smem_1 + R * STRIPE_W;
    bf16*  h_out_smem = silu_smem + BB * R;
    float* rstd_smem  = reinterpret_cast<float*>(h_out_smem + BB * STRIPE_W);
    uint64_t* mbar    = reinterpret_cast<uint64_t*>(
                            ((uintptr_t)(rstd_smem + BB) + 15) & ~(uintptr_t)15);
    // Individually-named mbars — no array indexing.
    uint64_t* const mbar_h_0 = &mbar[0];
    uint64_t* const mbar_h_1 = &mbar[1];
    uint64_t* const mbar_W_0 = &mbar[2];
    uint64_t* const mbar_W_1 = &mbar[3];

    constexpr float INV_D = 1.0f / float(D);
    constexpr float EPS   = 1e-6f;
    constexpr uint32_t H_STRIPE_BYTES = BB * STRIPE_W * 2;
    constexpr uint32_t W_STRIPE_BYTES = R * STRIPE_W * 2;

    if (tid == 0) {
        mbar_init(mbar_h_0, 1);
        mbar_init(mbar_h_1, 1);
        mbar_init(mbar_W_0, 1);
        mbar_init(mbar_W_1, 1);
    }
    __syncthreads();

    // ============ persistent block loop ============
    // Each CTA cycles through blocks (cta, cta+grid, cta+2*grid, ...).
    for (int block = cta; block < num_blocks; block += grid) {
        const int row0 = block * BB;

        // Copy X -> O for this block's rows.
        {
            constexpr int TOTAL = BB * D * 2 / 16;
            #pragma unroll 4
            for (int i = 0; i < TOTAL / NTHREADS; ++i) {
                int u = tid + i * NTHREADS;
                int r = u / (D / 8);
                int c = (u % (D / 8)) * 8;
                const int4* src = reinterpret_cast<const int4*>(X + (size_t)(row0 + r) * D + c);
                int4* dst = reinterpret_cast<int4*>(O + (size_t)(row0 + r) * D + c);
                *dst = *src;
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async;" ::: "memory");
        __syncthreads();

        // Per-buffer parities (reset each block — buffers are reused fresh).
        uint32_t p_h_0 = 0, p_h_1 = 0;
        uint32_t p_w_0 = 0, p_w_1 = 0;

    for (int l = 0; l < L; ++l) {
        //================================================================
        // Phase 1: V matmul (WG0) + fold h² (ALL warps) — parallel work
        //================================================================
        float z_acc[32];    // only WG0 uses this
        #pragma unroll
        for (int i = 0; i < 32; ++i) z_acc[i] = 0.f;

        constexpr int ROWS_PER_WARP = BB / NWARPS;   // 8 (with NWARPS=8)
        float sq_partial[ROWS_PER_WARP] = {};

        // Prime: issue TMA for stripe 0 into buffer 0.
        if (tid == 0) {
            mbar_arrive_expect_tx(mbar_h_0, H_STRIPE_BYTES);
            tma_load_2d(h_smem_0, &O_load_desc, 0, row0, mbar_h_0);
            mbar_arrive_expect_tx(mbar_W_0, W_STRIPE_BYTES);
            tma_load_2d(W_smem_0, &V_desc, 0, l * R, mbar_W_0);
        }

        for (int s = 0; s < N_STRIPES_D; ++s) {
            const bool cur_is_0 = ((s & 1) == 0);
            bf16* const     h_cur    = cur_is_0 ? h_smem_0 : h_smem_1;
            bf16* const     W_cur    = cur_is_0 ? W_smem_0 : W_smem_1;
            bf16* const     h_nxt    = cur_is_0 ? h_smem_1 : h_smem_0;
            bf16* const     W_nxt    = cur_is_0 ? W_smem_1 : W_smem_0;
            uint64_t* const mbar_h_c = cur_is_0 ? mbar_h_0 : mbar_h_1;
            uint64_t* const mbar_W_c = cur_is_0 ? mbar_W_0 : mbar_W_1;
            uint64_t* const mbar_h_n = cur_is_0 ? mbar_h_1 : mbar_h_0;
            uint64_t* const mbar_W_n = cur_is_0 ? mbar_W_1 : mbar_W_0;

            // Prefetch next stripe into buffer nxt (if not last).
            if (tid == 0 && s + 1 < N_STRIPES_D) {
                mbar_arrive_expect_tx(mbar_h_n, H_STRIPE_BYTES);
                tma_load_2d(h_nxt, &O_load_desc,
                            (s + 1) * STRIPE_W, row0, mbar_h_n);
                mbar_arrive_expect_tx(mbar_W_n, W_STRIPE_BYTES);
                tma_load_2d(W_nxt, &V_desc,
                            (s + 1) * STRIPE_W, l * R, mbar_W_n);
            }

            // Wait for THIS stripe's TMAs (buffer cur) to complete.
            uint32_t p_h_c = cur_is_0 ? p_h_0 : p_h_1;
            uint32_t p_w_c = cur_is_0 ? p_w_0 : p_w_1;
            mbar_wait(mbar_h_c, p_h_c);
            mbar_wait(mbar_W_c, p_w_c);
            if (cur_is_0) { p_h_0 ^= 1; p_w_0 ^= 1; }
            else          { p_h_1 ^= 1; p_w_1 ^= 1; }

            // Fold sq — ALL 8 warps participate (each owns 8 rows).
            // Runs in parallel with WG0's WGMMA below (different pipes).
            #pragma unroll
            for (int rw = 0; rw < ROWS_PER_WARP; ++rw) {
                int b = warp_id * ROWS_PER_WARP + rw;
                float partial = 0.f;
                #pragma unroll
                for (int d = lane; d < STRIPE_W; d += 32) {
                    float v = __bfloat162float(h_cur[hs_off(b, d)]);
                    partial += v * v;
                }
                sq_partial[rw] += partial;
            }

            // WGMMA V — warpgroup-scoped, only WG0 (128 threads) issues.
            if (wg_id == 0) {
                wgmma_fence();
                #pragma unroll
                for (int kt = 0; kt < K_TILES_STRIPE; ++kt) {
                    int koff = kt * WGMMA_K * 2;
                    bf16* a = (bf16*)((char*)h_cur + koff);
                    bf16* b = (bf16*)((char*)W_cur + koff);
                    uint64_t da = make_desc_sw128b(a, 64, 1);
                    uint64_t db = make_desc_sw128b(b, 64, 1);
                    int sd = (s == 0 && kt == 0) ? 0 : 1;
                    wgmma_m64n64k16(da, db, z_acc, sd, /*trans_b=*/0);
                }
                wgmma_commit();
                wgmma_wait<0>();
            }
            __syncthreads();   // hold both WGs before next stripe's TMA
        }

        // Reduce sq_partial across lanes (all 8 warps write to rstd_smem).
        #pragma unroll
        for (int rw = 0; rw < ROWS_PER_WARP; ++rw) {
            float sq = sq_partial[rw];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(~0u, sq, o);
            if (lane == 0) {
                int b = warp_id * ROWS_PER_WARP + rw;
                rstd_smem[b] = rsqrtf(sq * INV_D + EPS);
            }
        }
        __syncthreads();

        //================================================================
        // Phase 1.5: silu = SiLU(z * rstd), store silu_smem
        // Only WG0 has z_acc — use wg_warp for row assignment (0..3 → rows 0..63).
        //================================================================
        if (wg_id == 0) {
            const int row_base = wg_warp * 16;
            #pragma unroll
            for (int cg = 0; cg < 4; ++cg) {
                int c0 = tg * 2 + cg * 16;
                int r0 = row_base + gid;
                int r1 = row_base + gid + 8;
                float rr0 = rstd_smem[r0], rr1 = rstd_smem[r1];
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int col = c0 + half * 8;
                    float z00 = z_acc[cg*8 + half*4 + 0] * rr0;
                    float z01 = z_acc[cg*8 + half*4 + 1] * rr0;
                    float z10 = z_acc[cg*8 + half*4 + 2] * rr1;
                    float z11 = z_acc[cg*8 + half*4 + 3] * rr1;
                    float s00 = z00 / (1.f + __expf(-z00));
                    float s01 = z01 / (1.f + __expf(-z01));
                    float s10 = z10 / (1.f + __expf(-z10));
                    float s11 = z11 / (1.f + __expf(-z11));
                    bf162 p0 = __float22bfloat162_rn({s00, s01});
                    bf162 p1 = __float22bfloat162_rn({s10, s11});
                    *reinterpret_cast<bf162*>(&silu_smem[silu_off(r0, col)]) = p0;
                    *reinterpret_cast<bf162*>(&silu_smem[silu_off(r1, col)]) = p1;
                }
            }
        }
        __syncthreads();

        //================================================================
        // Phase 2: U matmul + h stripe update (double-buffered)
        //================================================================
        // Prime: issue TMA for stripe 0 into buffer 0.
        if (tid == 0) {
            mbar_arrive_expect_tx(mbar_h_0, H_STRIPE_BYTES);
            tma_load_2d(h_smem_0, &O_load_desc, 0, row0, mbar_h_0);
            mbar_arrive_expect_tx(mbar_W_0, W_STRIPE_BYTES);
            tma_load_2d(W_smem_0, &U_desc, 0, l * R, mbar_W_0);
        }

        for (int s = 0; s < N_STRIPES_D; ++s) {
            const bool cur_is_0 = ((s & 1) == 0);
            bf16* const     h_cur    = cur_is_0 ? h_smem_0 : h_smem_1;
            bf16* const     W_cur    = cur_is_0 ? W_smem_0 : W_smem_1;
            bf16* const     h_nxt    = cur_is_0 ? h_smem_1 : h_smem_0;
            bf16* const     W_nxt    = cur_is_0 ? W_smem_1 : W_smem_0;
            uint64_t* const mbar_h_c = cur_is_0 ? mbar_h_0 : mbar_h_1;
            uint64_t* const mbar_W_c = cur_is_0 ? mbar_W_0 : mbar_W_1;
            uint64_t* const mbar_h_n = cur_is_0 ? mbar_h_1 : mbar_h_0;
            uint64_t* const mbar_W_n = cur_is_0 ? mbar_W_1 : mbar_W_0;

            // Prefetch next stripe into nxt.
            if (tid == 0 && s + 1 < N_STRIPES_D) {
                mbar_arrive_expect_tx(mbar_h_n, H_STRIPE_BYTES);
                tma_load_2d(h_nxt, &O_load_desc,
                            (s + 1) * STRIPE_W, row0, mbar_h_n);
                mbar_arrive_expect_tx(mbar_W_n, W_STRIPE_BYTES);
                tma_load_2d(W_nxt, &U_desc,
                            (s + 1) * STRIPE_W, l * R, mbar_W_n);
            }

            uint32_t p_h_c = cur_is_0 ? p_h_0 : p_h_1;
            uint32_t p_w_c = cur_is_0 ? p_w_0 : p_w_1;
            mbar_wait(mbar_h_c, p_h_c);
            mbar_wait(mbar_W_c, p_w_c);
            if (cur_is_0) { p_h_0 ^= 1; p_w_0 ^= 1; }
            else          { p_h_1 ^= 1; p_w_1 ^= 1; }

            // h-fused init from cur buffer's h_smem (WG0 only — owns dh_acc).
            float dh_acc[32];
            if (wg_id == 0) {
                const int row_base = wg_warp * 16;
                #pragma unroll
                for (int cg = 0; cg < 4; ++cg) {
                    int c0 = tg * 2 + cg * 16;
                    int r0 = row_base + gid;
                    int r1 = row_base + gid + 8;
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        int col = c0 + half * 8;
                        bf162 p0 = *reinterpret_cast<const bf162*>(&h_cur[hs_off(r0, col)]);
                        bf162 p1 = *reinterpret_cast<const bf162*>(&h_cur[hs_off(r1, col)]);
                        dh_acc[cg*8 + half*4 + 0] = __low2float(p0);
                        dh_acc[cg*8 + half*4 + 1] = __high2float(p0);
                        dh_acc[cg*8 + half*4 + 2] = __low2float(p1);
                        dh_acc[cg*8 + half*4 + 3] = __high2float(p1);
                    }
                }

                // WGMMA U on current buffer's W_smem.
                wgmma_fence();
                #pragma unroll
                for (int kt = 0; kt < R / WGMMA_K; ++kt) {
                    int koff_a = kt * WGMMA_K * 2;
                    int koff_b = kt * WGMMA_K * STRIPE_W * 2;
                    bf16* a = (bf16*)((char*)silu_smem + koff_a);
                    bf16* b = (bf16*)((char*)W_cur + koff_b);
                    uint64_t da = make_desc_sw128b(a, 64, 1);
                    uint64_t db = make_desc_sw128b(b, 64, 1);
                    wgmma_m64n64k16(da, db, dh_acc, /*scale_d=*/1, /*trans_b=*/1);
                }
                wgmma_commit();
                wgmma_wait<0>();

                // Convert dh_acc fp32 → bf16, write to h_out_smem.
                const int row_base2 = wg_warp * 16;
                #pragma unroll
                for (int cg = 0; cg < 4; ++cg) {
                    int c0 = tg * 2 + cg * 16;
                    int r0 = row_base2 + gid;
                    int r1 = row_base2 + gid + 8;
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        int col = c0 + half * 8;
                        bf162 p0 = __float22bfloat162_rn({dh_acc[cg*8 + half*4 + 0],
                                                          dh_acc[cg*8 + half*4 + 1]});
                        bf162 p1 = __float22bfloat162_rn({dh_acc[cg*8 + half*4 + 2],
                                                          dh_acc[cg*8 + half*4 + 3]});
                        *reinterpret_cast<bf162*>(&h_out_smem[hs_off(r0, col)]) = p0;
                        *reinterpret_cast<bf162*>(&h_out_smem[hs_off(r1, col)]) = p1;
                    }
                }
            }
            __syncthreads();  // both WGs sync before proxy fence + TMA store
            // Regular SMEM writes → async TMA-store proxy visibility fence
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

            // TMA store h_out_smem → O[stripe s].
            if (tid == 0) {
                tma_store_2d(h_out_smem, &O_store_desc, s * STRIPE_W, row0);
                tma_store_commit();
                tma_store_wait<0>();
            }
            __syncthreads();
        }
        asm volatile("fence.proxy.async;" ::: "memory");
        __syncthreads();
    }
    }   // end persistent block loop
}

static CUtensorMap make_tma_desc_2d_bb64(const bf16* gmem, int rows, int cols) {
    CUtensorMap desc{};
    cuuint64_t size[2]   = { (cuuint64_t)cols, (cuuint64_t)rows };
    cuuint64_t stride[1] = { (cuuint64_t)(cols * 2) };
    cuuint32_t box[2]    = { (cuuint32_t)STRIPE_W, (cuuint32_t)BB };
    cuuint32_t elem_str[2] = { 1, 1 };
    CUresult res = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<bf16*>(gmem), size, stride, box, elem_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled (bb64) failed: ", int(res));
    return desc;
}
static CUtensorMap make_tma_desc_2d_R(const bf16* gmem, int rows, int cols) {
    CUtensorMap desc{};
    cuuint64_t size[2]   = { (cuuint64_t)cols, (cuuint64_t)rows };
    cuuint64_t stride[1] = { (cuuint64_t)(cols * 2) };
    cuuint32_t box[2]    = { (cuuint32_t)STRIPE_W, (cuuint32_t)R };
    cuuint32_t elem_str[2] = { 1, 1 };
    CUresult res = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<bf16*>(gmem), size, stride, box, elem_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled (R) failed: ", int(res));
    return desc;
}

torch::Tensor dlrnet_wgmma_persistent(
    torch::Tensor X, torch::Tensor Vs, torch::Tensor Us)
{
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16);
    int B = X.size(0);
    TORCH_CHECK(B % BB == 0, "B must be multiple of BB=64");
    TORCH_CHECK(X.size(1) == D);
    int L = Vs.size(0);
    TORCH_CHECK(Vs.size(1) == R && Vs.size(2) == D);
    TORCH_CHECK(Us.size(1) == R && Us.size(2) == D);

    auto O = torch::empty_like(X);
    auto Xp = reinterpret_cast<const bf16*>(X.const_data_ptr<at::BFloat16>());
    auto Op = reinterpret_cast<bf16*>(O.data_ptr<at::BFloat16>());
    auto Vp = reinterpret_cast<const bf16*>(Vs.const_data_ptr<at::BFloat16>());
    auto Up = reinterpret_cast<const bf16*>(Us.const_data_ptr<at::BFloat16>());

    CUtensorMap V_desc       = make_tma_desc_2d_R(Vp, L * R, D);
    CUtensorMap U_desc       = make_tma_desc_2d_R(Up, L * R, D);
    CUtensorMap O_load_desc  = make_tma_desc_2d_bb64(Op, B, D);
    CUtensorMap O_store_desc = make_tma_desc_2d_bb64(Op, B, D);

    // Dynamic SMEM: double-buffered h and W stripes, silu scratch, h_out
    // stripe, per-row rstd, and the mbarriers.
    int smem_bytes = BB * STRIPE_W * 2 * 2   // h_smem[2]
                   + R  * STRIPE_W * 2 * 2   // W_smem[2]
                   + BB * R * 2              // silu_smem
                   + BB * STRIPE_W * 2       // h_out_smem
                   + BB * 4                  // rstd_smem
                   + 64                      // mbarriers + padding
                   + 256;
    cudaError_t err = cudaFuncSetAttribute(
        dlr_h100_prefill_17b_v30_balanced_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    TORCH_CHECK(err == cudaSuccess);

    int num_blocks = B / BB;

    // Balanced-wave persistent grid. This kernel is memory-bandwidth bound, so
    // a partial/sparse final wave wastes bandwidth-time. Instead of launching
    // one CTA per row-block, minimize the number of hardware waves and split
    // work evenly so every wave is equally full:
    //   waves = ceil(num_blocks / R_max)
    //   grid  = ceil(num_blocks / waves)   (every CTA does ~equal blocks)
    // The persistent stride loop handles any grid, so correctness is unchanged.
    int dev;
    cudaGetDevice(&dev);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, dlr_h100_prefill_17b_v30_balanced_kernel,
        NTHREADS, smem_bytes);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    int r_max = blocks_per_sm * sm_count;          // max resident CTAs (e.g. 264)
    int waves = (num_blocks + r_max - 1) / r_max;  // ceil: minimum # of HW passes
    int grid  = (num_blocks + waves - 1) / waves;  // ceil: even split across waves

    dlr_h100_prefill_17b_v30_balanced_kernel<<<grid, NTHREADS, smem_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        V_desc, U_desc, O_load_desc, O_store_desc, Xp, Op, L, num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlrnet_wgmma_persistent",
          &dlrnet_wgmma_persistent,
          "1.7B DLR-Net WGMMA persistent balanced-wave prefill");
}
