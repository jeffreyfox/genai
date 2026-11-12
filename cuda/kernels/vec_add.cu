#include "kernels/vec_add.cuh"
#include "common/timing.cuh"

#include <cuda_runtime.h>

__global__ void vectorAddKernel(float *A, float *B, float *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n, int n_threads, int warmup, int iters) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    int n_blocks = (n + n_threads - 1) / n_threads;
    dim3 grid(n_blocks);
    dim3 block(n_threads);
    auto launch = [=]() {
        vectorAddKernel<<<grid, block>>>(A_d, B_d, C_d, n);
    };
    time_kernel_launches(
        launch, n, grid, block, warmup, iters, 3.0 * size, static_cast<double>(n));

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
