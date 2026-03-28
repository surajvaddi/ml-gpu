#pragma once

#include <cstddef>
#include <string>

namespace linear_regression
{

    // ================================
    // Backend enumeration
    // ================================

    enum class Backend
    {
        CPU,
        GPU_A_NAIVE,
        GPU_B_OPTIMIZED,
        GPU_C_FUSED,
        GPU_CUBLAS
    };

    // ================================
    // Configuration
    // ================================

    struct Config
    {
        int num_iters = 100;
        float learning_rate = 1e-2f;

        // Elastic Net compatibility
        float l1 = 0.0f; // Lasso
        float l2 = 0.0f; // Ridge
    };

    // ================================
    // Training interface (Unified)
    // ================================

    /*
     * X: row-major matrix of shape (n_samples, n_features)
     * y: vector length n_samples
     * w: vector length n_features (updated in-place)
     */
    void train(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg,
        Backend backend);

    // ================================
    // Explicit backend entrypoints
    // ================================

    void train_cpu(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg);

    void train_gpu_A_naive(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg);

    void train_gpu_B_optimized(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg);

    void train_gpu_C_fused(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg);

    void train_gpu_cublas(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg);

    // ================================
    // Loss computation (shared reference)
    // ================================

    float compute_mse(
        const float *X,
        const float *y,
        const float *w,
        int n_samples,
        int n_features);

    // ================================
    // Benchmark timing structure
    // ================================

    struct Timing
    {
        double total_ms = 0.0;
        double h2d_ms = 0.0;
        double kernel_ms = 0.0;
        double d2h_ms = 0.0;
    };

} // namespace linear_regression