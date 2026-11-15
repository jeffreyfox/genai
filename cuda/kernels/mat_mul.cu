#include "kernels/mat_mul.cuh"
#include "common/timing.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

// A: m x k
// B: k x n
// C: m x n  
__global__ void matMulKernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < m && col < n) {
        float value = 0.0f;
        for (int inner = 0; inner < k; ++inner) {
            // A[row][inner] * B[inner][col]
            value += A[row * k + inner] * B[inner * n + col];
        }
        C[row * n + col] = value;
    }
}

// tiled matrix multiplication kernel
// A: m x k
// B: k x n
// C: m x n
template <int TILE_WIDTH>
__global__ void matMulTiledKernel(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float value = 0.0f;
    for (int phase = 0; phase < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
        // copy A[row][phase*TILE_WIDTH+tx] to Ads
        if (row < m && phase*TILE_WIDTH+tx < k) {
            Ads[ty][tx] = A[row*k + phase*TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0.0f;
        }
        // copy B[phase*TILE_WIDTH+ty][col] to Bds
        if (phase*TILE_WIDTH+ty < k && col < n) {
            Bds[ty][tx] = B[(phase*TILE_WIDTH+ty)*n + col];
        } else {
            Bds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int inner = 0; inner < TILE_WIDTH; ++inner) {
            value += Ads[ty][inner] * Bds[inner][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        C[row * n + col] = value;
    }
}

void matMul(float* A_h, float* B_h, float* C_h, int m, int n, int k, int block_size, bool tiled) {
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("m, n, and k must be positive");
    }
    if (block_size < 1 || block_size > 32) {
        throw std::invalid_argument("block_size must be in [1, 32]");
    }
    if (tiled && block_size != 4 && block_size != 8 && block_size != 16 && block_size != 32) {
        throw std::invalid_argument("block_size must be 4, 8, 16, or 32");
    }
    size_t size_A = (size_t)m * (size_t)k * sizeof(float);
    size_t size_B = (size_t)k * (size_t)n * sizeof(float);
    size_t size_C = (size_t)m * (size_t)n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size_A);
    cudaMalloc((void **) &B_d, size_B);
    cudaMalloc((void **) &C_d, size_C);

    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    int n_blocks_y = (m + block_size - 1) / block_size;
    int n_blocks_x = (n + block_size - 1) / block_size;
    dim3 grid(n_blocks_x, n_blocks_y);
    dim3 block(block_size, block_size);
    if (tiled) {
        auto launch = [=]() {
            switch (block_size) {
                case 4:
                    matMulTiledKernel<4><<<grid, block>>>(A_d, B_d, C_d, m, n, k);
                    break;
                case 8:
                    matMulTiledKernel<8><<<grid, block>>>(A_d, B_d, C_d, m, n, k);
                    break;
                case 16:
                    matMulTiledKernel<16><<<grid, block>>>(A_d, B_d, C_d, m, n, k);
                    break;
                case 32:
                    matMulTiledKernel<32><<<grid, block>>>(A_d, B_d, C_d, m, n, k);
                    break;
                default:
                    throw std::invalid_argument("block_size must be 4, 8, 16, or 32");
            }
        };
        auto flops_per_launch = 2.0 * m * n * k;
        time_kernel_launches(
            launch, m, n, k, grid, block, flops_per_launch);    
    } else {
        auto launch = [=]() {
            matMulKernel<<<grid, block>>>(A_d, B_d, C_d, m, n, k);
        };
        auto flops_per_launch = 2.0 * m * n * k;
        time_kernel_launches(
            launch, m, n, k, grid, block, flops_per_launch);
    }

    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


// void matMul2(float* A_h, float* B_h, float* C_h, int m, int n, int k, int n_threads, bool tiled, int tile_width) {
//     // if (tiled && tile_width != 4 && tile_width != 8 && tile_width != 16 && tile_width != 32) {
//     //     throw std::invalid_argument("tile_width must be 4, 8, 16, or 32");
//     // }
//     size_t size_A = (size_t)m * (size_t)k * sizeof(float);
//     size_t size_B = (size_t)k * (size_t)n * sizeof(float);
//     size_t size_C = (size_t)m * (size_t)n * sizeof(float);
//     float *A_d, *B_d, *C_d;

//     cudaMalloc((void **) &A_d, size_A);
//     cudaMalloc((void **) &B_d, size_B);
//     cudaMalloc((void **) &C_d, size_C);

//     cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
//     cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

//     int block_width = tiled ? tile_width : n_threads;
//     int n_blocks = (width + block_width - 1) / block_width;
//     dim3 grid(n_blocks, n_blocks);
//     dim3 block(block_width, block_width);
//     if (tiled) {
//         auto launch = [=]() {
//             switch (tile_width) {
//                 case 4:
//                     matMulTiledKernel<4><<<grid, block>>>(A_d, B_d, C_d, width);
//                     break;
//                 case 8:
//                     matMulTiledKernel<8><<<grid, block>>>(A_d, B_d, C_d, width);
//                     break;
//                 case 16:
//                     matMulTiledKernel<16><<<grid, block>>>(A_d, B_d, C_d, width);
//                     break;
//                 case 32:
//                     matMulTiledKernel<32><<<grid, block>>>(A_d, B_d, C_d, width);
//                     break;
//                 default:
//                     throw std::invalid_argument("tile_width must be 4, 8, 16, or 32");
//             }
//         };
//         auto flops_per_launch = 2.0 * width * width * width;
//         time_kernel_launches(
//             launch, width, grid, block, flops_per_launch);    
//     } else {
//         auto launch = [=]() {
//             matMulKernel<<<grid, block>>>(A_d, B_d, C_d, width);
//         };
//         auto flops_per_launch = 2.0 * width * width * width;
//         time_kernel_launches(
//             launch, width, grid, block, flops_per_launch);
//     }

//     cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

//     cudaFree(A_d);
//     cudaFree(B_d);
//     cudaFree(C_d);
// }
