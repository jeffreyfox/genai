#include "common/verify.cuh"
#include "kernels/mat_mul.cuh"
#include "kernels/vec_add.cuh"
#include "kernels/sum_red.cuh"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // ./bench [--kernel vec_add|mat_mul] [n] [n_threads] [--verify]
    int positional[2];
    int npos = 0;
    bool do_verify = false;
    bool tiled = false;
    int block_size = -1;
    int m = -1, k = -1;
    std::string kernel = "vec_add";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 || std::strcmp(argv[i], "--k") == 0) {
            const bool is_m = std::strcmp(argv[i], "--m") == 0;
            if (i + 1 >= argc) {
                std::cerr << "--m and --k require a positive dimension\n";
                return 1;
            }
            int dimension = std::atoi(argv[++i]);
            if (dimension <= 0) {
                std::cerr << "--m and --k require a positive dimension\n";
                return 1;
            }
            (is_m ? m : k) = dimension;
            continue;
        }
        if (std::strcmp(argv[i], "--tiled") == 0) {
            tiled = true;
            continue;
        }
        if (std::strcmp(argv[i], "--block-size") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "--block-size requires a side length\n";
                return 1;
            }
            block_size = std::atoi(argv[++i]);
            continue;
        }
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
        if (npos < 2) {
            positional[npos++] = std::atoi(argv[i]);
        } else {
            std::cerr << "expected at most n and n_threads as positional arguments\n";
            return 1;
        }
    }

    const bool is_mat_mul = (kernel == "mat_mul");
    if (!is_mat_mul && kernel != "vec_add" && kernel != "sum_red") {
        std::cerr << "unknown kernel: " << kernel << " (expected vec_add, mat_mul, or sum_red)\n";
        return 1;
    }

    int n = npos > 0 ? positional[0] : (is_mat_mul ? 256 : (1 << 20));
    if (!is_mat_mul && (m != -1 || k != -1)) {
        std::cerr << "--m and --k require --kernel mat_mul\n";
        return 1;
    }
    if (m == -1) m = n;
    if (k == -1) k = n;
    int n_threads = npos > 1 ? positional[1] : (is_mat_mul ? 32 : 256);
    if (block_size == -1) {
        block_size = n_threads;
    }

    if (n <= 0) {
        std::cerr << "n must be positive\n";
        return 1;
    }
    if (n_threads < 1 || n_threads > 1024) {
        std::cerr << "n_threads must be in [1, 1024]\n";
        return 1;
    }
    if (is_mat_mul && (block_size < 1 || block_size > 32)) {
        std::cerr << "mat_mul block_size must be in [1, 32]\n";
        return 1;
    }
    if (tiled && (!is_mat_mul || (block_size != 4 && block_size != 8 && block_size != 16 && block_size != 32))) {
        std::cerr << "--tiled requires mat_mul and --block-size 4, 8, 16, or 32\n";
        return 1;
    }

    const size_t a_elems = is_mat_mul ? (size_t)m * (size_t)k : (size_t)n;
    const size_t b_elems = is_mat_mul ? (size_t)k * (size_t)n : (size_t)n;
    const size_t c_elems = is_mat_mul ? (size_t)m * (size_t)n : (size_t)n;
    std::vector<float> A_h(a_elems), B_h(b_elems), C_h(c_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (float& value : A_h) value = dis(gen);
    for (float& value : B_h) value = dis(gen);

    if (is_mat_mul) {
        matMul(A_h.data(), B_h.data(), C_h.data(), m, n, k, block_size, tiled);
        if (do_verify && !verify_mat_mul(A_h.data(), B_h.data(), C_h.data(), m, n, k)) {
            return 1;
        }
    } else if (kernel == "sum_red") {
        std::vector<float> original = A_h;
        sumRed(A_h.data(), n);
        if (do_verify && !verify_sum(original.data(), A_h[0], n)) {
            return 1;
        }
    } else {
        vecAdd(A_h.data(), B_h.data(), C_h.data(), n, n_threads);
        if (do_verify && !verify_vec_add(A_h.data(), B_h.data(), C_h.data(), n)) {
            return 1;
        }
    }
    return 0;
}
