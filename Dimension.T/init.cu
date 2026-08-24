// matrix_mul_shared.cu - Tiled Matrix Multiplication using Shared Memory
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

// CUDA Kernel: C = A * B for N x N matrices
__global__ void matrixMulShared(const float *A, const float *B, float *C, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify row and column of target C element
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Allocate shared memory tiles for block collaboration
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    float value = 0.0f;

    // Loop over sub-matrix tiles along the internal dimension
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        
        // 1. Collaborative load from global memory to shared memory (with bounds check)
        if (row < N && (m * TILE_SIZE + tx) < N) {
            s_A[ty][tx] = A[row * N + (m * TILE_SIZE + tx)];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if ((m * TILE_SIZE + ty) < N && col < N) {
            s_B[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // 2. Wait until all threads in block finish loading the tile
        __syncthreads();

        // 3. Multiply sub-tile out of shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += s_A[ty][k] * s_B[k][tx];
        }

        // 4. Synchronize before next tile overwrites shared buffers
        __syncthreads();
    }

    // Write final accumulated result to global VRAM
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure 2D grid and 2D block execution layout
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Verification (C[0] expected " << 2.0f * N << "): " << h_C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
