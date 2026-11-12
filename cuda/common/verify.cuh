#pragma once

#include <cmath>
#include <iostream>

inline bool verify_vec_add(const float* A_h, const float* B_h, const float* C_h, int n, float tol = 1e-6f) {
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

inline bool verify_mat_mul(
    const float* M_h, const float* N_h, const float* P_h, int width,
    float atol = 1e-3f, float rtol = 1e-3f) {
    int mismatches = 0;
    float max_abs_err = 0.0f;
    int first_bad = -1;
    float first_got = 0.0f;
    float first_want = 0.0f;

    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float want = 0.0f;
            for (int k = 0; k < width; ++k) {
                want += M_h[row * width + k] * N_h[k * width + col];
            }
            float got = P_h[row * width + col];
            float err = fabsf(got - want);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
            float thresh = atol + rtol * fabsf(want);
            if (err > thresh) {
                if (mismatches == 0) {
                    first_bad = row * width + col;
                    first_got = got;
                    first_want = want;
                }
                ++mismatches;
            }
        }
    }

    if (mismatches == 0) {
        std::cerr << "verify=pass max_abs_err=" << max_abs_err
                  << " atol=" << atol << " rtol=" << rtol << std::endl;
        return true;
    }

    std::cerr << "verify=fail mismatches=" << mismatches
              << " max_abs_err=" << max_abs_err
              << " first_i=" << first_bad
              << " got=" << first_got
              << " want=" << first_want
              << " atol=" << atol << " rtol=" << rtol << std::endl;
    return false;
}
