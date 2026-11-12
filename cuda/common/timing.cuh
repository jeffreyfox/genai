#pragma once

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

inline void print_dim3(dim3 d) {
    std::cout << d.x << 'x' << d.y << 'x' << d.z;
}

template <typename Launch>
void time_kernel_launches(
    Launch launch,
    int n,
    dim3 grid,
    dim3 block,
    int warmup,
    int iters,
    double bytes_per_launch,
    double flops_per_launch) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) {
        launch();
    }

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    float ms = ms_total / iters;
    double bandwidth_GBs = bytes_per_launch / (ms * 1.0e6);
    double gflops_s = flops_per_launch / (ms * 1.0e6);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << prop.name << '\t'
              << n << '\t';
    print_dim3(grid);
    std::cout << '\t';
    print_dim3(block);
    std::cout << '\t'
              << warmup << '\t'
              << iters << '\t'
              << std::scientific << std::setprecision(4)
              << ms << '\t'
              << bandwidth_GBs << '\t'
              << gflops_s
              << std::defaultfloat
              << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
