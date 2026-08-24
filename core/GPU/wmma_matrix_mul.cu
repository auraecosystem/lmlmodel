// wmma_matrix_mul.cu - Tensor Core Matrix Multiplication using CUDA WMMA
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA tile dimensions (16x16x16 FP16 tile is hardware-standard for Tensor Cores)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel: C (float) = A (half) * B (half)
__global__ void wmmaGemm(const __half *A, const __half *B, float *C, int M, int N, int K) {
    // 128 threads per block = 4 warps (32 threads per warp)
    // Map block layout: 2 warps along Y (rows), 2 warps along X (cols) -> 32x32 output per block
    int warpIdx = threadIdx.x / 32;
    int warpRow = (blockIdx.y * 2) + (warpIdx / 2);
    int warpCol = (blockIdx.x * 2) + (warpIdx % 2);

    // Bounds check
    if (warpRow * WMMA_M >= M || warpCol * WMMA_N >= N) return;

    // 1. Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 2. Initialize accumulator tile to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // 3. Loop over the K dimension in steps of WMMA_K (16)
    for (int k = 0; k < K; k += WMMA_K) {
        const half *a_ptr = A + (warpRow * WMMA_M) * K + k;
        const half *b_ptr = B + k * N + (warpCol * WMMA_N);

        // Load 16x16 sub-matrices into warp registers
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);

        // Perform Tensor Core hardware matrix multiply-accumulate across the warp
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4. Write output fragment back to global VRAM
    float *c_ptr = C + (warpRow * WMMA_M) * N + (warpCol * WMMA_N);
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

int main() {
    int M = 512, N = 512, K = 512;

    size_t size_A = M * K * sizeof(__half);
    size_t size_B = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(float);

    // Host allocation
    __half *h_A = (__half *)malloc(size_A);
    __half *h_B = (__half *)malloc(size_B);
    float  *h_C = (float *)malloc(size_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(2.0f);

    // Device allocation
    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Execution configuration (4 warps = 128 threads per block)
    dim3 threadsPerBlock(128, 1, 1);
    dim3 blocksPerGrid((N + 31) / 32, (M + 31) / 32, 1);

    wmmaGemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "Verification (C[0] expected " << 2.0f * K << "): " << h_C[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
