#include "kernels/mat_mul.cuh"
#include "common/timing.cuh"

#include <cuda_runtime.h>

__global__ void matMulKernel(float *M, float *N, float *P, int width) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < width && col < width) {
        float value = 0.0f;
        for (int k = 0; k < width; ++k) {
            value += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = value;
    }
}

void matMul(float* M_h, float* N_h, float* P_h, int width, int n_threads, int warmup, int iters) {
    size_t size = (size_t)width * (size_t)width * sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **) &M_d, size);
    cudaMalloc((void **) &N_d, size);
    cudaMalloc((void **) &P_d, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    int n_blocks = (width + n_threads - 1) / n_threads;
    dim3 grid(n_blocks, n_blocks);
    dim3 block(n_threads, n_threads);
    auto launch = [=]() {
        matMulKernel<<<grid, block>>>(M_d, N_d, P_d, width);
    };
    auto bytes_per_launch = (2.0 * width * width * width + 1.0 * width * width) * sizeof(float);
    auto flops_per_launch = 2.0 * width * width * width;
    time_kernel_launches(
        launch, width, grid, block, warmup, iters, bytes_per_launch, flops_per_launch);

    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}
