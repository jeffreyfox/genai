#include "kernels/sum_red.cuh"
#include "common/timing.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

__global__ void sumRedKernel(float *input, float *output, size_t n) {
    size_t i = 2 * size_t(threadIdx.x);
    for (size_t stride = 1; stride < n; stride *= 2) {
        if (threadIdx.x % stride == 0 && i + stride < n) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = n == 0 ? 0.0f : input[0];
    }
}

void sumRed(float* input_h, int n) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }

    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    unsigned int n_threads = (static_cast<unsigned int>(n) + 1) / 2;
    std::cerr << "device=" << prop.name
              << ", maxThreadsPerBlock=" << prop.maxThreadsPerBlock
              << ", maxThreadsDim[0]=" << prop.maxThreadsDim[0]
              << ", n_threads=" << n_threads << std::endl;
    if (n_threads > prop.maxThreadsPerBlock || n_threads > prop.maxThreadsDim[0]) {
        throw std::invalid_argument("input is too large for a single block");
    }

    size_t size = size_t(n) * sizeof(float);
    float *input_d, *original_d, *output_d;
    cudaMalloc((void **) &input_d, size);
    cudaMalloc((void **) &original_d, size);
    cudaMalloc((void **) &output_d, sizeof(float));
    cudaMemcpy(original_d, input_h, size, cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(n_threads);
    auto launch = [=]() {
        // Restore the input because each reduction modifies it in place.
        cudaMemcpyAsync(input_d, original_d, size, cudaMemcpyDeviceToDevice);
        sumRedKernel<<<grid, block>>>(input_d, output_d, n);
    };
    time_kernel_launches(launch, 1, n, 1, grid, block, double(n - 1));

    cudaMemcpy(input_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(original_d);
    cudaFree(output_d);
}
