#include "../../include/linear_regression.hpp"
#include "../../../common/cuda/cuda_utils.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <cassert>

namespace linear_regression
{

    // ========================================
    // Kernel 1: Compute gradient (atomic)
    // ========================================

    __global__ void compute_gradient_naive(
        const float *X,
        const float *y,
        const float *w,
        float *grad,
        int n_samples,
        int n_features)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_samples)
            return;

        const float *Xi = &X[i * n_features];

        // Compute prediction
        float pred = 0.0f;
        for (int j = 0; j < n_features; ++j)
            pred += Xi[j] * w[j];

        float residual = pred - y[i];

        // Accumulate gradient (X^T r)
        for (int j = 0; j < n_features; ++j)
            atomicAdd(&grad[j], Xi[j] * residual);
    }

    // ========================================
    // Kernel 2: Weight update
    // ========================================

    __global__ void update_weights(
        float *w,
        float *grad,
        int n_features,
        float lr_over_n,
        float l1,
        float l2)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= n_features)
            return;

        float g = grad[j];

        // L2 regularization
        if (l2 != 0.0f)
            g += l2 * w[j];

        // L1 regularization
        if (l1 != 0.0f)
        {
            if (w[j] > 0.0f)
                g += l1;
            else if (w[j] < 0.0f)
                g -= l1;
        }

        w[j] -= lr_over_n * g;
    }

    // ========================================
    // GPU-A Training Function
    // ========================================

    void train_gpu_A_naive(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg)
    {
        size_t X_size = n_samples * n_features * sizeof(float);
        size_t y_size = n_samples * sizeof(float);
        size_t w_size = n_features * sizeof(float);
        size_t grad_size = n_features * sizeof(float);

        float *d_X, *d_y, *d_w, *d_grad;

        cudaMalloc(&d_X, X_size);
        cudaMalloc(&d_y, y_size);
        cudaMalloc(&d_w, w_size);
        cudaMalloc(&d_grad, grad_size);

        cudaMemcpy(d_X, X, X_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, y_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks_samples = (n_samples + threads - 1) / threads;
        int blocks_features = (n_features + threads - 1) / threads;

        float lr_over_n = cfg.learning_rate / n_samples;

        for (int iter = 0; iter < cfg.num_iters; ++iter)
        {

            cudaMemset(d_grad, 0, grad_size);

            compute_gradient_naive<<<blocks_samples, threads>>>(
                d_X, d_y, d_w, d_grad,
                n_samples, n_features);

            update_weights<<<blocks_features, threads>>>(
                d_w, d_grad,
                n_features,
                lr_over_n,
                cfg.l1,
                cfg.l2);

            cudaDeviceSynchronize();
        }

        cudaMemcpy(w, d_w, w_size, cudaMemcpyDeviceToHost);

        cudaFree(d_X);
        cudaFree(d_y);
        cudaFree(d_w);
        cudaFree(d_grad);
    }

} // namespace linear_regression