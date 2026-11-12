#include "common/verify.cuh"
#include "kernels/mat_mul.cuh"
#include "kernels/vec_add.cuh"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // ./bench [--kernel vec_add|mat_mul] [n] [n_threads] [iters] [warmup] [--verify]
    int positional[4];
    int npos = 0;
    bool do_verify = false;
    std::string kernel = "vec_add";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0) {
            do_verify = true;
            continue;
        }
        if (std::strcmp(argv[i], "--kernel") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "--kernel requires a name (vec_add or mat_mul)\n";
                return 1;
            }
            kernel = argv[++i];
            continue;
        }
        if (npos < 4) {
            positional[npos++] = std::atoi(argv[i]);
        }
    }

    const bool is_mat_mul = (kernel == "mat_mul");
    if (!is_mat_mul && kernel != "vec_add") {
        std::cerr << "unknown kernel: " << kernel << " (expected vec_add or mat_mul)\n";
        return 1;
    }

    int n = npos > 0 ? positional[0] : (is_mat_mul ? 256 : (1 << 20));
    int n_threads = npos > 1 ? positional[1] : (is_mat_mul ? 32 : 256);
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
    if (is_mat_mul && n_threads * n_threads > 1024) {
        std::cerr << "mat_mul uses n_threads x n_threads per block; n_threads must be <= 32\n";
        return 1;
    }
    if (iters <= 0 || warmup < 0) {
        std::cerr << "iters must be > 0 and warmup >= 0\n";
        return 1;
    }

    const size_t n_elems = is_mat_mul ? (size_t)n * (size_t)n : (size_t)n;
    std::vector<float> A_h(n_elems), B_h(n_elems), C_h(n_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (size_t i = 0; i < n_elems; ++i) {
        A_h[i] = dis(gen);
        B_h[i] = dis(gen);
    }

    if (is_mat_mul) {
        matMul(A_h.data(), B_h.data(), C_h.data(), n, n_threads, warmup, iters);
        if (do_verify && !verify_mat_mul(A_h.data(), B_h.data(), C_h.data(), n)) {
            return 1;
        }
    } else {
        vecAdd(A_h.data(), B_h.data(), C_h.data(), n, n_threads, warmup, iters);
        if (do_verify && !verify_vec_add(A_h.data(), B_h.data(), C_h.data(), n)) {
            return 1;
        }
    }
    return 0;
}
