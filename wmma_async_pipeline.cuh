// wmma_async_pipeline.cu - Pipelined Tensor Core GEMM with cp.async (sm_80+)
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// PTX Wrapper: Asynchronously copy 16 bytes directly from Global to Shared memory
__device__ __forceinline__ void cp_async_16bytes(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

// Group outstanding cp.async instructions into a logical commit batch
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until N or fewer commit groups remain in-flight
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ void wmmaAsyncPipelinedGemm(const half *A, const half *B, float *C, int M, int N, int K) {
    // 1. Double-buffered shared memory stages [stage][row][col]
    __shared__ half s_A[2][WMMA_M][WMMA_K];
    __shared__ half s_B[2][WMMA_K][WMMA_N];

    int warpIdx = threadIdx.x / 32;
    int warpRow = (blockIdx.y * 2) + (warpIdx / 2);
    int warpCol = (blockIdx.x * 2) + (warpIdx % 2);

    if (warpRow * WMMA_M >= M || warpCol * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Lambda helper: Distribute 16-byte async transfers across threads in block
    auto issue_async_tile_load = [&](int buffer_stage, int k_offset) {
        int tid = threadIdx.x;

        // Load s_A (256 elements = 32 transfers of 8 halves / 16 bytes)
        if (tid < 32) {
            int row = tid / 2;
            int col = (tid % 2) * 8;
            const half* g_ptr = A + (warpRow * WMMA_M + row) * K + (k_offset + col);
            cp_async_16bytes(&s_A[buffer_stage][row][col], g_ptr);
        }
        // Load s_B (256 elements = 32 transfers of 8 halves / 16 bytes)
        if (tid >= 32 && tid < 64) {
            int t_offset = tid - 32;
            int row = t_offset / 2;
            int col = (t_offset % 2) * 8;
            const half* g_ptr = B + (k_offset + row) * N + (warpCol * WMMA_N + col);
            cp_async_16bytes(&s_B[buffer_stage][row][col], g_ptr);
        }
    };

    // --- PROLOGUE: Prefetch tile 0 into Stage 0 ---
    issue_async_tile_load(0, 0);
    cp_async_commit();

    int read_stage = 0;
    int write_stage = 1;

    // --- PIPELINED MAIN LOOP ---
    for (int k = 0; k < K; k += WMMA_K) {
        int next_k = k + WMMA_K;

        // Step A: Issue async fetch for NEXT tile into write_stage buffer
        if (next_k < K) {
            issue_async_tile_load(write_stage, next_k);
            cp_async_commit();
        }

        // Step B: Wait for CURRENT tile in read_stage to complete transfer
        if (next_k < K) {
            cp_async_wait<1>(); // Keep 1 group in-flight (the next tile loading)
        } else {
            cp_async_wait<0>(); // Final iteration: wait for all groups
        }
        __syncthreads();

        // Step C: Execute WMMA on CURRENT tile while NEXT tile transfers in background
        wmma::load_matrix_sync(a_frag, &s_A[read_stage][0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &s_B[read_stage][0][0], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();

        // Step D: Swap buffer stages (0 -> 1 -> 0)
        read_stage ^= 1;
        write_stage ^= 1;
    }

    // Write final result to VRAM
    float *c_ptr = C + (warpRow * WMMA_M) * N + (warpCol * WMMA_N);
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}
