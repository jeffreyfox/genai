#include "kernels/mat_mul.cuh"
#include "common/timing.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

__global__ void matMulKernel(float *A, float *B, float *C, int width) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < width && col < width) {
        float value = 0.0f;
        for (int k = 0; k < width; ++k) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

// tiled matrix multiplication kernel
template <int TILE_WIDTH>
__global__ void matMulTiledKernel(float *A, float *B, float *C, int width) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float value = 0.0f;
    for (int phase = 0; phase < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
        // copy A[row][phase*TILE_WIDTH+tx] to Ads
        if (row < width && phase*TILE_WIDTH+tx < width) {
            Ads[ty][tx] = A[row*width + phase*TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0.0f;
        }
        // copy B[phase*TILE_WIDTH+ty][col] to Bds
        if (phase*TILE_WIDTH+ty < width && col < width) {
            Bds[ty][tx] = B[(phase*TILE_WIDTH+ty)*width + col];
        } else {
            Bds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    if (row < width && col < width) {
        C[row * width + col] = value;
    }
}

void matMul(float* A_h, float* B_h, float* C_h, int width, int n_threads, int warmup, int iters, bool tiled, int tile_width) {
    if (tiled && tile_width != 4 && tile_width != 8 && tile_width != 16 && tile_width != 32) {
        throw std::invalid_argument("tile_width must be 4, 8, 16, or 32");
    }
    size_t size = (size_t)width * (size_t)width * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    int block_width = tiled ? tile_width : n_threads;
    int n_blocks = (width + block_width - 1) / block_width;
    dim3 grid(n_blocks, n_blocks);
    dim3 block(block_width, block_width);
    if (tiled) {
        auto launch = [=]() {
            switch (tile_width) {
                case 4:
                    matMulTiledKernel<4><<<grid, block>>>(A_d, B_d, C_d, width);
                    break;
                case 8:
                    matMulTiledKernel<8><<<grid, block>>>(A_d, B_d, C_d, width);
                    break;
                case 16:
                    matMulTiledKernel<16><<<grid, block>>>(A_d, B_d, C_d, width);
                    break;
                case 32:
                    matMulTiledKernel<32><<<grid, block>>>(A_d, B_d, C_d, width);
                    break;
                default:
                    throw std::invalid_argument("tile_width must be 4, 8, 16, or 32");
            }
        };
        auto bytes_per_launch = (2.0 * width * width * n_blocks + 1.0 * width * width) * sizeof(float);
        auto flops_per_launch = 2.0 * width * width * width;
        time_kernel_launches(
            launch, width, grid, block, warmup, iters, bytes_per_launch, flops_per_launch);    
    } else {
        auto launch = [=]() {
            matMulKernel<<<grid, block>>>(A_d, B_d, C_d, width);
        };
        auto bytes_per_launch = (2.0 * width * width * width + 1.0 * width * width) * sizeof(float);
        auto flops_per_launch = 2.0 * width * width * width;
        time_kernel_launches(
            launch, width, grid, block, warmup, iters, bytes_per_launch, flops_per_launch);
    }

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
