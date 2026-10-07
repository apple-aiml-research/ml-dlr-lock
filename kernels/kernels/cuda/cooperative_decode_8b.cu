//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Cooperative fused DLR-Net decode kernel (D=4096, R=128), small-batch.
//
// Runs one forward pass of a stack of L DLR-Net residual layers, each computing
//   h += U @ silu((h @ V) * rstd),   rstd = 1/sqrt(mean(h^2) + eps),
// with the RMSNorm gain folded into V and the residual scale folded into U. This
// variant targets the small-batch decode regime: the grid is (NP, B), so one
// group of NP cooperating CTAs handles a single row. Each CTA owns one 1/NP
// stripe of the D dimension, computing its share of the V projection and its
// share of the U output. Per layer, atomic-spin barriers combine the partial
// results across the NP CTAs through global workspace buffers (H_ws, SZ_ws). The
// U weight tile for each layer is staged into shared memory with cp.async at the
// top of the layer so the load overlaps the V matmul and the barrier wait.
//
// Compile-time config: D=CFG_D, R=CFG_R, NP=CFG_NP, BLOCK_K=CFG_BLOCK_K,
// NWARPS=CFG_NWARPS, FENCE=CFG_FENCE.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;

#ifndef CFG_D
#define CFG_D 4096
#endif
#ifndef CFG_R
#define CFG_R 128
#endif
#ifndef CFG_NP
#define CFG_NP 32
#endif
#ifndef CFG_BLOCK_K
#define CFG_BLOCK_K 512
#endif
#ifndef CFG_NWARPS
#define CFG_NWARPS 4
#endif
#ifndef CFG_FENCE
#define CFG_FENCE 1
#endif

constexpr int D = CFG_D;
constexpr int R = CFG_R;
constexpr int NUM_PROGRAMS = CFG_NP;
constexpr int NWARPS = CFG_NWARPS;
constexpr int NTHREADS = NWARPS * 32;
constexpr int BLOCK_K = CFG_BLOCK_K;
constexpr int NUM_K = D / BLOCK_K;
constexpr int R_PER_CTA = R / NUM_PROGRAMS;
constexpr int Z_PER_WARP = R_PER_CTA / NWARPS;
constexpr int H_PER_THREAD = D / NTHREADS;
constexpr int U_CHUNK = D / NUM_PROGRAMS;
constexpr int VEC = 8;
constexpr int N_VEC_PER_LANE = BLOCK_K / (32 * VEC);

// U_smem cp.async: each thread issues U_CPA_PER_THREAD int4 (16B = 8 bf16) ops.
// Total bytes = R * U_CHUNK * 2; per-thread = total / (NTHREADS * 16).
constexpr int U_BYTES = R * U_CHUNK * 2;
constexpr int U_CPA_PER_THREAD = U_BYTES / (NTHREADS * 16);
static_assert(U_BYTES == NTHREADS * 16 * U_CPA_PER_THREAD, "U_BYTES must divide cleanly");

static_assert(R_PER_CTA * NUM_PROGRAMS == R, "R must be divisible by NP");
static_assert(Z_PER_WARP * NWARPS == R_PER_CTA, "R_PER_CTA must be divisible by NWARPS");

__device__ __forceinline__ void cp_async_16(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_ptr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
dlr_h100_multi_prog_v13_multib_kernel(
    const bf16* __restrict__ X,
    bf16*       __restrict__ O,
    bf16*       __restrict__ H_ws,
    bf16*       __restrict__ SZ_ws,
    int*        __restrict__ BAR,
    const bf16* __restrict__ Vs,
    const bf16* __restrict__ Us,
    int L
) {
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int cta_id = blockIdx.x;
    const int row    = blockIdx.y;
    const int u_start = cta_id * U_CHUNK;
    const int r_start = cta_id * R_PER_CTA;

    bf16* H_row   = H_ws  + (size_t)row * D;
    bf16* SZ_row  = SZ_ws + (size_t)row * R;
    int*  BAR_row = BAR   + (size_t)row * (1 + L * 2);
    const bf16* X_row = X + (size_t)row * D;
    bf16* O_row = O + (size_t)row * D;

    __shared__ float warp_sums[NWARPS];
    __shared__ float sz_smem[R];
    __shared__ float h_smem[D];
    extern __shared__ bf16  U_smem[];   // 32 KB (dynamic)

    constexpr float INV_D = 1.0f / float(D);
    constexpr float EPS = 1e-6f;

    // ---- Init: CTA 0 of this row copies X_row → H_row ----
    if (cta_id == 0) {
        #pragma unroll
        for (int i = 0; i < H_PER_THREAD; ++i)
            H_row[tid + i * NTHREADS] = X_row[tid + i * NTHREADS];
    }
    if (tid == 0) atomicAdd(&BAR_row[0], 1);
    __syncthreads();
    if (tid == 0) { while (atomicAdd(&BAR_row[0], 0) < NUM_PROGRAMS) {} }
    __syncthreads();

    const uint32_t U_smem_base = __cvta_generic_to_shared(U_smem);

    for (int l = 0; l < L; ++l) {
        const bf16* V_l = Vs + (size_t)l * R * D;
        const bf16* U_l = Us + (size_t)l * R * D;

        // ---- Issue cp.async for U_smem (overlap with V matmul + barrier wait) ----
        // Layout: U_smem[r, c] mirrors U_l[r, u_start + c]. Each thread loads
        // U_CPA_PER_THREAD * 8 bf16 = U_CPA_PER_THREAD int4 ops, strided by NTHREADS.
        #pragma unroll
        for (int j = 0; j < U_CPA_PER_THREAD; ++j) {
            int idx = j * NTHREADS + tid;               // linear bf16 index in U_smem
            int rr = idx / (U_CHUNK / VEC);             // row r
            int cc = (idx % (U_CHUNK / VEC)) * VEC;     // col c (multiple of 8)
            const void* gmem = &U_l[(size_t)rr * D + u_start + cc];
            uint32_t smem = U_smem_base + (rr * U_CHUNK + cc) * sizeof(bf16);
            cp_async_16(smem, gmem);
        }
        cp_async_commit();

        // ---- RMSNorm + h*rstd into h_smem ----
        float h_reg[H_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < H_PER_THREAD; ++i)
            h_reg[i] = __bfloat162float(H_row[tid + i * NTHREADS]);

        float sq = 0.f;
        #pragma unroll
        for (int i = 0; i < H_PER_THREAD; ++i) sq += h_reg[i] * h_reg[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(~0u, sq, o);
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane < NWARPS) ? warp_sums[lane] : 0.f;
            #pragma unroll
            for (int o = NWARPS / 2; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
            if (lane == 0) warp_sums[0] = v;
        }
        __syncthreads();
        float rstd = rsqrtf(warp_sums[0] * INV_D + EPS);

        #pragma unroll
        for (int i = 0; i < H_PER_THREAD; ++i)
            h_smem[tid + i * NTHREADS] = h_reg[i] * rstd;
        __syncthreads();

        // ---- V matmul ----
        float z_local[Z_PER_WARP];
        #pragma unroll
        for (int i = 0; i < Z_PER_WARP; ++i) z_local[i] = 0.f;

        #pragma unroll
        for (int kc = 0; kc < NUM_K; ++kc) {
            int k_base = kc * BLOCK_K;
            #pragma unroll
            for (int ii = 0; ii < N_VEC_PER_LANE; ++ii) {
                int d_local = lane * VEC + ii * (32 * VEC);
                int d = k_base + d_local;
                float h_arr[VEC];
                #pragma unroll
                for (int k = 0; k < VEC; ++k) h_arr[k] = h_smem[d + k];
                #pragma unroll
                for (int zz = 0; zz < Z_PER_WARP; ++zz) {
                    int r = r_start + warp_id * Z_PER_WARP + zz;
                    int4 v_vec = *reinterpret_cast<const int4*>(&V_l[r * D + d]);
                    const bf16* v_bf = reinterpret_cast<const bf16*>(&v_vec);
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k)
                        z_local[zz] += h_arr[k] * __bfloat162float(v_bf[k]);
                }
            }
        }

        #pragma unroll
        for (int zz = 0; zz < Z_PER_WARP; ++zz) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                z_local[zz] += __shfl_xor_sync(~0u, z_local[zz], o);
        }

        if (lane == 0) {
            #pragma unroll
            for (int zz = 0; zz < Z_PER_WARP; ++zz) {
                int r = r_start + warp_id * Z_PER_WARP + zz;
                float z = z_local[zz];
                float sig = 1.0f / (1.0f + __expf(-z));
                SZ_row[r] = __float2bfloat16(z * sig);
            }
        }

        // ---- Phase-1 barrier ----
        __syncthreads();
        #if CFG_FENCE
        __threadfence();
        #endif
        if (tid == 0) atomicAdd(&BAR_row[1 + l * 2], 1);
        if (tid == 0) {
            volatile int* bar_ptr = &BAR_row[1 + l * 2];
            while (*bar_ptr < NUM_PROGRAMS) {}
        }
        __syncthreads();

        // ---- Reload sz_smem ----
        if (tid < R) sz_smem[tid] = __bfloat162float(SZ_row[tid]);

        // ---- Wait for U_smem load (issued at top, overlapped with V) ----
        cp_async_wait_all();
        __syncthreads();

        // ---- U matmul (reads U_smem instead of __ldg) ----
        for (int d = tid; d < U_CHUNK; d += NTHREADS) {
            float dh = 0.f;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float sz_r = sz_smem[r];
                float u_val = __bfloat162float(U_smem[r * U_CHUNK + d]);
                dh += sz_r * u_val;
            }
            int d_global = u_start + d;
            float h_old = __bfloat162float(H_row[d_global]);
            H_row[d_global] = __float2bfloat16(h_old + dh);
        }

        // ---- Phase-2 barrier ----
        __syncthreads();
        #if CFG_FENCE
        __threadfence();
        #endif
        if (tid == 0) atomicAdd(&BAR_row[1 + l * 2 + 1], 1);
        if (tid == 0) {
            volatile int* bar_ptr = &BAR_row[1 + l * 2 + 1];
            while (*bar_ptr < NUM_PROGRAMS) {}
        }
        __syncthreads();
    }

    if (cta_id == 0) {
        #pragma unroll
        for (int i = 0; i < H_PER_THREAD; ++i)
            O_row[tid + i * NTHREADS] = H_row[tid + i * NTHREADS];
    }
}

torch::Tensor dlrnet_cooperative_decode_8b(torch::Tensor X, torch::Tensor Vs, torch::Tensor Us) {
    int B = X.size(0);
    int L = Vs.size(0);
    auto O = torch::empty_like(X);
    auto H_ws  = torch::zeros({(int64_t)B, D}, X.options().dtype(torch::kBFloat16));
    auto SZ_ws = torch::zeros({(int64_t)B, R}, X.options().dtype(torch::kBFloat16));
    auto BAR   = torch::zeros({(int64_t)B * (1 + L * 2)}, X.options().dtype(torch::kInt32));

    auto Xp  = reinterpret_cast<const bf16*>(X.const_data_ptr<at::BFloat16>());
    auto Op  = reinterpret_cast<bf16*>(O.data_ptr<at::BFloat16>());
    auto Hp  = reinterpret_cast<bf16*>(H_ws.data_ptr<at::BFloat16>());
    auto SZp = reinterpret_cast<bf16*>(SZ_ws.data_ptr<at::BFloat16>());
    auto Vp  = reinterpret_cast<const bf16*>(Vs.const_data_ptr<at::BFloat16>());
    auto Up  = reinterpret_cast<const bf16*>(Us.const_data_ptr<at::BFloat16>());

    dim3 grid(NUM_PROGRAMS, B);
    int smem_bytes = U_BYTES;
    cudaFuncSetAttribute(
        dlr_h100_multi_prog_v13_multib_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);
    dlr_h100_multi_prog_v13_multib_kernel<<<grid, NTHREADS, smem_bytes,
                                            at::cuda::getCurrentCUDAStream()>>>(
        Xp, Op, Hp, SZp, BAR.data_ptr<int>(), Vp, Up, L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlrnet_cooperative_decode_8b", &dlrnet_cooperative_decode_8b,
          "Cooperative fused DLR-Net small-batch decode (D=4096, R=128)");
}
