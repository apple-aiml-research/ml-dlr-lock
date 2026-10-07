//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Fused DLR-Net prefill kernel with WGMMA + TMA (D=1024, R=32).
//
// Each DLR-Net layer computes h += alpha * U @ silu(V @ rmsnorm(h)), where the
// per-channel RMS gain is folded into V and alpha is folded into U, so the
// kernel evaluates z = (h @ V) * rstd, sz = silu(z), dh = sz @ U, h += dh. The
// V and U matmuls both run on Hopper WGMMA tensor cores; weight tiles are
// streamed in via TMA (cp.async.bulk.tensor). Warp groups are specialized so a
// matmul overlaps with the RMSNorm reduction and the SiLU activation with the
// next weight load, and the residual add is folded into the U accumulator. The
// activations h stay resident in shared memory across all layers.
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
#define CFG_D 1024
#endif
#ifndef CFG_R
#define CFG_R 32
#endif
#ifndef CFG_BB
#define CFG_BB 64
#endif

constexpr int D = CFG_D;
constexpr int R = CFG_R;
constexpr int BB = CFG_BB;
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;
constexpr int WG_SIZE = 4;

constexpr int WGMMA_M   = 64;
constexpr int WGMMA_N_V = 32;
constexpr int WGMMA_K   = 16;
constexpr int STRIPE_W  = 64;
constexpr int N_STRIPES_D = D / STRIPE_W;
constexpr int K_TILES_PER_STRIPE = STRIPE_W / WGMMA_K;

constexpr int WGMMA_N_U = 64;
constexpr int N_TILES_U = D / WGMMA_N_U;
constexpr int K_TILES_U = R / WGMMA_K;
constexpr int N_TILES_PER_WG = N_TILES_U / 2;

constexpr int TMA_BOX_COLS = STRIPE_W;
constexpr int N_STRIPES_W  = N_STRIPES_D;
constexpr uint32_t MBAR_BYTES = R * D * 2;

static_assert(BB == WGMMA_M, "");
static_assert(R == WGMMA_N_V, "");
static_assert(N_TILES_U % 2 == 0, "");

__device__ __forceinline__ int sw128(int row, int col) {
    int tile_idx = row >> 3, row_in_tile = row & 7;
    int core_col = col >> 3, sw_col = core_col ^ row_in_tile;
    int elem = col & 7;
    return tile_idx * 512 + row_in_tile * 64 + sw_col * 8 + elem;
}
__device__ __forceinline__ int hsm_off(int row, int col) {
    int stripe = col >> 6, col_in_stripe = col & 63;
    return stripe * (BB * 64) + sw128(row, col_in_stripe);
}
__device__ __forceinline__ int silu_off(int row, int col) { return sw128(row, col); }
__device__ __forceinline__ uint32_t to_smem(const void* p) { return __cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = to_smem(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

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
        "{\n"
        ".reg .pred p;\n"
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
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void
wgmma_m64n32k16_V(uint64_t desc_a, uint64_t desc_b, float* d, int scale_d) {
    asm volatile(
        "{\n .reg .pred p; setp.ne.b32 p, %18, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16"
        " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
        " %16, %17, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

__device__ __forceinline__ void
wgmma_m64n64k16_U(uint64_t desc_a, uint64_t desc_b, float* d, int scale_d) {
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

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
dlrnet_wgmma_vu_tma_v4_06b_kernel(
    const __grid_constant__ CUtensorMap V_desc,
    const __grid_constant__ CUtensorMap U_desc,
    const bf16* __restrict__ X,
    bf16*       __restrict__ O,
    int L)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tg = lane & 3;
    const int wg_id = warp_id >> 2;
    const int wg_warp = warp_id & 3;
    const int cta = blockIdx.x, row0 = cta * BB;

    extern __shared__ __align__(16) char smem_raw[];
    bf16*  h_smem    = reinterpret_cast<bf16*>(smem_raw);
    bf16*  W_smem    = h_smem + BB * D;
    bf16*  silu_smem = W_smem + R * D;
    float* rstd_smem = reinterpret_cast<float*>(silu_smem + BB * STRIPE_W);
    uint64_t* mbar_V = reinterpret_cast<uint64_t*>(
        ((uintptr_t)(rstd_smem + BB) + 15) & ~(uintptr_t)15);
    uint64_t* mbar_U = mbar_V + 1;

    constexpr float INV_D = 1.0f / float(D);
    constexpr float EPS = 1e-6f;

    if (tid == 0) {
        mbar_init(mbar_V, 1);
        mbar_init(mbar_U, 1);
    }
    __syncthreads();

    {
        constexpr int H_VEC = BB * D * 2 / 16;
        #pragma unroll 8
        for (int i = 0; i < H_VEC / NTHREADS; ++i) {
            int u = tid + i * NTHREADS;
            int r = u / (D / 8), c = (u % (D / 8)) * 8;
            cp_async_16(h_smem + hsm_off(r, c), X + (row0 + r) * D + c);
        }
        cp_async_commit();
        cp_async_wait<0>();
    }
    __syncthreads();

    uint32_t parity_V = 0, parity_U = 0;

    for (int l = 0; l < L; ++l) {
        if (tid == 0) mbar_arrive_expect_tx(mbar_V, MBAR_BYTES);
        if (tid < N_STRIPES_W) {
            int stripe = tid;
            int col = stripe * TMA_BOX_COLS;
            int row = l * R;
            bf16* dst = W_smem + stripe * (R * TMA_BOX_COLS);
            tma_load_2d(dst, &V_desc, col, row, mbar_V);
        }

        if (wg_id == 1) {
            const int r_base = wg_warp * (BB / WG_SIZE);
            #pragma unroll
            for (int rw = 0; rw < BB / WG_SIZE; ++rw) {
                int row = r_base + rw;
                float sq = 0.f;
                #pragma unroll
                for (int d = lane; d < D; d += 32) {
                    float v = __bfloat162float(h_smem[hsm_off(row, d)]);
                    sq += v * v;
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(~0u, sq, o);
                if (lane == 0) rstd_smem[row] = rsqrtf(sq * INV_D + EPS);
            }
        }

        mbar_wait(mbar_V, parity_V);
        parity_V ^= 1;

        float z_acc[16];
        if (wg_id == 0) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) z_acc[i] = 0.f;

            wgmma_fence();
            #pragma unroll
            for (int stripe = 0; stripe < N_STRIPES_D; ++stripe) {
                bf16* a_sb = h_smem + stripe * (BB * STRIPE_W);
                bf16* b_sb = W_smem + stripe * (R  * STRIPE_W);
                #pragma unroll
                for (int kt = 0; kt < K_TILES_PER_STRIPE; ++kt) {
                    int koff = kt * WGMMA_K * 2;
                    bf16* a = (bf16*)((char*)a_sb + koff);
                    bf16* b = (bf16*)((char*)b_sb + koff);
                    uint64_t da = make_desc_sw128b(a, 64, 1);
                    uint64_t db = make_desc_sw128b(b, 64, 1);
                    int sd = (stripe == 0 && kt == 0) ? 0 : 1;
                    wgmma_m64n32k16_V(da, db, z_acc, sd);
                }
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        __syncthreads();

        // SiLU on the V-matmul result, written to silu_smem as packed bf16 pairs.
        if (wg_id == 0) {
            const int row_base = wg_warp * 16;
            #pragma unroll
            for (int cp = 0; cp < R / 8; ++cp) {
                int r0 = row_base + gid, r1 = row_base + gid + 8;
                int ca = cp * 8 + tg * 2 + 0;  // c0 for the pair
                float rr0 = rstd_smem[r0], rr1 = rstd_smem[r1];
                float z00 = z_acc[cp*4+0]*rr0, z01 = z_acc[cp*4+1]*rr0;
                float z10 = z_acc[cp*4+2]*rr1, z11 = z_acc[cp*4+3]*rr1;
                float s00 = z00/(1.f+__expf(-z00)), s01 = z01/(1.f+__expf(-z01));
                float s10 = z10/(1.f+__expf(-z10)), s11 = z11/(1.f+__expf(-z11));
                // Packed bf16 store — 2 conversions per pair
                bf162 pair0 = __float22bfloat162_rn({s00, s01});
                bf162 pair1 = __float22bfloat162_rn({s10, s11});
                *reinterpret_cast<bf162*>(&silu_smem[silu_off(r0, ca)]) = pair0;
                *reinterpret_cast<bf162*>(&silu_smem[silu_off(r1, ca)]) = pair1;
            }
        } else {
            if (tid == 128) mbar_arrive_expect_tx(mbar_U, MBAR_BYTES);
            int wg1_local = tid - 128;
            if (wg1_local < N_STRIPES_W) {
                int stripe = wg1_local;
                int col = stripe * TMA_BOX_COLS;
                int row = l * R;
                bf16* dst = W_smem + stripe * (R * TMA_BOX_COLS);
                tma_load_2d(dst, &U_desc, col, row, mbar_U);
            }
        }
        __syncthreads();
        mbar_wait(mbar_U, parity_U);
        parity_U ^= 1;

        // WGMMA U matmul with the residual add folded into the accumulator:
        // load current h into the accumulator, then dh = silu(z) @ U adds on top.
        {
            const int nt_start = wg_id * N_TILES_PER_WG;
            const int nt_end   = nt_start + N_TILES_PER_WG;
            const int row_base = wg_warp * 16;

            #pragma unroll 1
            for (int nt = nt_start; nt < nt_end; ++nt) {
                int n_base = nt * WGMMA_N_U;
                bf16* b_stripe_base = W_smem + nt * (R * STRIPE_W);

                float dh_acc[32];
                // Packed load: h[r, c0..c1] read as bf162, split into 2 fp32
                #pragma unroll
                for (int col_pair = 0; col_pair < WGMMA_N_U / 8; ++col_pair) {
                    int r0 = row_base + gid;
                    int r1 = row_base + gid + 8;
                    int col_a = col_pair * 8 + tg * 2 + 0;
                    int c0 = n_base + col_a;
                    int idx = col_pair * 4;
                    bf162 pair0 = *reinterpret_cast<const bf162*>(&h_smem[hsm_off(r0, c0)]);
                    bf162 pair1 = *reinterpret_cast<const bf162*>(&h_smem[hsm_off(r1, c0)]);
                    dh_acc[idx+0] = __low2float(pair0);
                    dh_acc[idx+1] = __high2float(pair0);
                    dh_acc[idx+2] = __low2float(pair1);
                    dh_acc[idx+3] = __high2float(pair1);
                }

                wgmma_fence();
                #pragma unroll
                for (int kt = 0; kt < K_TILES_U; ++kt) {
                    bf16* a_base = (bf16*)((char*)silu_smem + kt * WGMMA_K * 2);
                    bf16* b_base = b_stripe_base + kt * WGMMA_K * STRIPE_W;
                    uint64_t desc_a = make_desc_sw128b(a_base, 64, 1);
                    uint64_t desc_b = make_desc_sw128b(b_base, 64, 1);
                    wgmma_m64n64k16_U(desc_a, desc_b, dh_acc, /*scale_d=*/1);
                }
                wgmma_commit();
                wgmma_wait<0>();

                // Packed store: pair of fp32 → bf162, single 4-byte store
                #pragma unroll
                for (int col_pair = 0; col_pair < WGMMA_N_U / 8; ++col_pair) {
                    int r0 = row_base + gid;
                    int r1 = row_base + gid + 8;
                    int col_a = col_pair * 8 + tg * 2 + 0;
                    int c0 = n_base + col_a;
                    int idx = col_pair * 4;
                    bf162 pair0 = __float22bfloat162_rn({dh_acc[idx+0], dh_acc[idx+1]});
                    bf162 pair1 = __float22bfloat162_rn({dh_acc[idx+2], dh_acc[idx+3]});
                    *reinterpret_cast<bf162*>(&h_smem[hsm_off(r0, c0)]) = pair0;
                    *reinterpret_cast<bf162*>(&h_smem[hsm_off(r1, c0)]) = pair1;
                }
            }
        }
        __syncthreads();
    }

    {
        constexpr int VEC = BB * D * 2 / 16;
        #pragma unroll 8
        for (int i = 0; i < VEC / NTHREADS; ++i) {
            int u = tid + i * NTHREADS;
            int r = u / (D / 8), c = (u % (D / 8)) * 8;
            int4 v = *reinterpret_cast<const int4*>(h_smem + hsm_off(r, c));
            *reinterpret_cast<int4*>(O + (size_t)(row0 + r) * D + c) = v;
        }
    }
}

static CUtensorMap make_tma_desc_2d_wgmma(const bf16* gmem, int rows, int cols) {
    CUtensorMap desc{};
    cuuint64_t size[2]   = { (cuuint64_t)cols, (cuuint64_t)rows };
    cuuint64_t stride[1] = { (cuuint64_t)(cols * 2) };
    cuuint32_t box[2]    = { (cuuint32_t)TMA_BOX_COLS, (cuuint32_t)R };
    cuuint32_t elem_str[2] = { 1, 1 };
    CUresult res = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<bf16*>(gmem), size, stride, box, elem_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(res));
    return desc;
}

torch::Tensor dlrnet_wgmma_fused(torch::Tensor X, torch::Tensor Vs, torch::Tensor Us) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16);
    int B = X.size(0);
    TORCH_CHECK(B % BB == 0);
    TORCH_CHECK(X.size(1) == D);
    int L = Vs.size(0);
    TORCH_CHECK(Vs.size(1) == R && Vs.size(2) == D);
    TORCH_CHECK(Us.size(1) == R && Us.size(2) == D);

    auto O = torch::empty_like(X);
    auto Xp = reinterpret_cast<const bf16*>(X.const_data_ptr<at::BFloat16>());
    auto Op = reinterpret_cast<bf16*>(O.data_ptr<at::BFloat16>());
    auto Vp = reinterpret_cast<const bf16*>(Vs.const_data_ptr<at::BFloat16>());
    auto Up = reinterpret_cast<const bf16*>(Us.const_data_ptr<at::BFloat16>());

    CUtensorMap V_desc = make_tma_desc_2d_wgmma(Vp, L * R, D);
    CUtensorMap U_desc = make_tma_desc_2d_wgmma(Up, L * R, D);

    int smem_bytes = BB * D * 2 + R * D * 2 + BB * STRIPE_W * 2 + BB * 4
                   + 16 + 2 * 8 + 256;
    cudaError_t err = cudaFuncSetAttribute(dlrnet_wgmma_vu_tma_v4_06b_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    TORCH_CHECK(err == cudaSuccess);

    int grid = B / BB;
    dlrnet_wgmma_vu_tma_v4_06b_kernel<<<grid, NTHREADS, smem_bytes,
                                         at::cuda::getCurrentCUDAStream()>>>(
        V_desc, U_desc, Xp, Op, L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlrnet_wgmma_fused", &dlrnet_wgmma_fused,
          "Fused DLR-Net prefill (WGMMA + TMA, D=1024, R=32)");
}
