#include "kernels/mat_mul.cuh"
#include "common/timing.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

__global__ void matMulKernel(float *M, float *N, float *P, int width) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < width && col < width) {
        float PValue = 0.0f;
        for (int k = 0; k < width; ++k) {
            PValue += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = PValue;
    }
}

// tiled matrix multiplication kernel
template <int TILE_WIDTH>
__global__ void matMulTiledKernel(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float PValue = 0.0f;
    for (int phase = 0; phase < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
        // copy M[row][phase*TILE_WIDTH+tx] to Mds
        if (row < width && phase*TILE_WIDTH+tx < width) {
            Mds[ty][tx] = M[row*width + phase*TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }
        // copy N[phase*TILE_WIDTH+ty][col] to Nds
        if (phase*TILE_WIDTH+ty < width && col < width) {
            Nds[ty][tx] = N[(phase*TILE_WIDTH+ty)*width + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            PValue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (row < width && col < width) {
        P[row * width + col] = PValue;
    }
}

void matMul(float* M_h, float* N_h, float* P_h, int width, int n_threads, int warmup, int iters, bool tiled, int tile_width) {
    if (tiled && tile_width != 4 && tile_width != 8 && tile_width != 16 && tile_width != 32) {
        throw std::invalid_argument("tile_width must be 4, 8, 16, or 32");
    }
    size_t size = (size_t)width * (size_t)width * sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **) &M_d, size);
    cudaMalloc((void **) &N_d, size);
    cudaMalloc((void **) &P_d, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    int block_width = tiled ? tile_width : n_threads;
    int n_blocks = (width + block_width - 1) / block_width;
    dim3 grid(n_blocks, n_blocks);
    dim3 block(block_width, block_width);
    if (tiled) {
        auto launch = [=]() {
            switch (tile_width) {
                case 4:
                    matMulTiledKernel<4><<<grid, block>>>(M_d, N_d, P_d, width);
                    break;
                case 8:
                    matMulTiledKernel<8><<<grid, block>>>(M_d, N_d, P_d, width);
                    break;
                case 16:
                    matMulTiledKernel<16><<<grid, block>>>(M_d, N_d, P_d, width);
                    break;
                case 32:
                    matMulTiledKernel<32><<<grid, block>>>(M_d, N_d, P_d, width);
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
            matMulKernel<<<grid, block>>>(M_d, N_d, P_d, width);
        };
        auto bytes_per_launch = (2.0 * width * width * width + 1.0 * width * width) * sizeof(float);
        auto flops_per_launch = 2.0 * width * width * width;
        time_kernel_launches(
            launch, width, grid, block, warmup, iters, bytes_per_launch, flops_per_launch);
    }

    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}
