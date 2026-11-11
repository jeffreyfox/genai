#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n_blocks = (n + n_threads - 1) / n_threads;

    for (int i = 0; i < warmup; ++i) {
        vectorAddKernel<<<n_blocks, n_threads>>>(A_d, B_d, C_d, n);
    }

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        vectorAddKernel<<<n_blocks, n_threads>>>(A_d, B_d, C_d, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    float ms = ms_total / iters;
    // 2 reads + 1 write per element; 1 add per element
    double bandwidth_GBs = (3.0 * size) / (ms * 1.0e6);
    double gflops_s = n / (ms * 1.0e6);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << prop.name << '\t'
              << n << '\t'
              << n_threads << '\t'
              << n_blocks << '\t'
              << warmup << '\t'
              << iters << '\t'
              << std::scientific << std::setprecision(4)
              << ms << '\t'
              << bandwidth_GBs << '\t'
              << gflops_s
              << std::defaultfloat
              << std::endl;

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

bool verify(const float* A_h, const float* B_h, const float* C_h, int n, float tol = 1e-6f) {
    int mismatches = 0;
    float max_abs_err = 0.0f;
    int first_bad = -1;
    float first_got = 0.0f;
    float first_want = 0.0f;

    for (int i = 0; i < n; ++i) {
        float want = A_h[i] + B_h[i];
        float err = fabsf(C_h[i] - want);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        if (err > tol) {
            if (mismatches == 0) {
                first_bad = i;
                first_got = C_h[i];
                first_want = want;
            }
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::cerr << "verify=pass max_abs_err=" << max_abs_err << " tol=" << tol << std::endl;
        return true;
    }

    std::cerr << "verify=fail mismatches=" << mismatches
              << " max_abs_err=" << max_abs_err
              << " first_i=" << first_bad
              << " got=" << first_got
              << " want=" << first_want
              << " tol=" << tol << std::endl;
    return false;
}

int main(int argc, char** argv) {
    // ./vec_add [n] [n_threads] [iters] [warmup] [--verify]
    int positional[4];
    int npos = 0;
    bool do_verify = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0) {
            do_verify = true;
            continue;
        }
        if (npos < 4) {
            positional[npos++] = std::atoi(argv[i]);
        }
    }

    int n = npos > 0 ? positional[0] : (1 << 20);
    int n_threads = npos > 1 ? positional[1] : 256;
    int iters = npos > 2 ? positional[2] : 50;
    int warmup = npos > 3 ? positional[3] : 5;

    if (n <= 0) {
        std::cerr << "n must be positive\n";
        return 1;
    }
    if (n_threads < 1 || n_threads > 1024) {
        std::cerr << "n_threads must be in [1, 1024]\n";
        return 1;
    }
    if (iters <= 0 || warmup < 0) {
        std::cerr << "iters must be > 0 and warmup >= 0\n";
        return 1;
    }

    std::vector<float> A_h(n), B_h(n), C_h(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (int i = 0; i < n; ++i) {
        A_h[i] = dis(gen);
        B_h[i] = dis(gen);
    }

    vecAdd(A_h.data(), B_h.data(), C_h.data(), n, n_threads, warmup, iters);
    if (do_verify && !verify(A_h.data(), B_h.data(), C_h.data(), n)) {
        return 1;
    }
    return 0;
}
