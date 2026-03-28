#include "../../include/linear_regression.hpp"
#include "../../../common/cuda/cuda_utils.cuh"

#include <cuda_runtime.h>
#include <vector>

namespace linear_regression
{

    constexpr int BLOCK_SIZE = 256;

    // =======================================================
    // Kernel 1: Per-block gradient accumulation (no atomics)
    // =======================================================

    __global__ void compute_gradient_block(
        const float *X,
        const float *y,
        const float *w,
        float *block_grads,
        int n_samples,
        int n_features)
    {
        extern __shared__ float s_grad[];

        int tid = threadIdx.x;
        int sample = blockIdx.x * blockDim.x + tid;

        // Initialize shared gradient
        for (int j = tid; j < n_features; j += blockDim.x)
            s_grad[j] = 0.0f;

        __syncthreads();

        if (sample < n_samples)
        {
            const float *Xi = &X[sample * n_features];

            float pred = 0.0f;
            for (int j = 0; j < n_features; ++j)
                pred += Xi[j] * w[j];

            float residual = pred - y[sample];

            for (int j = 0; j < n_features; ++j)
                atomicAdd(&s_grad[j], Xi[j] * residual);
        }

        __syncthreads();

        // Write block result to global memory
        for (int j = tid; j < n_features; j += blockDim.x)
            block_grads[blockIdx.x * n_features + j] = s_grad[j];
    }

    // =======================================================
    // Kernel 2: Reduce block gradients
    // =======================================================

    __global__ void reduce_block_grads(
        float *block_grads,
        float *grad,
        int n_blocks,
        int n_features)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= n_features)
            return;

        float sum = 0.0f;
        for (int b = 0; b < n_blocks; ++b)
            sum += block_grads[b * n_features + j];

        grad[j] = sum;
    }

    // =======================================================
    // Kernel 3: Weight update (same logic as GPU-A)
    // =======================================================

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

        if (l2 != 0.0f)
            g += l2 * w[j];

        if (l1 != 0.0f)
        {
            if (w[j] > 0.0f)
                g += l1;
            else if (w[j] < 0.0f)
                g -= l1;
        }

        w[j] -= lr_over_n * g;
    }

    // =======================================================
    // GPU-B Training Function
    // =======================================================

    void train_gpu_B_optimized(
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

        int threads = BLOCK_SIZE;
        int blocks = (n_samples + threads - 1) / threads;

        size_t block_grad_size = blocks * n_features * sizeof(float);

        float *d_X, *d_y, *d_w;
        float *d_grad, *d_block_grads;

        cudaMalloc(&d_X, X_size);
        cudaMalloc(&d_y, y_size);
        cudaMalloc(&d_w, w_size);
        cudaMalloc(&d_grad, grad_size);
        cudaMalloc(&d_block_grads, block_grad_size);

        cudaMemcpy(d_X, X, X_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, y_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice);

        float lr_over_n = cfg.learning_rate / n_samples;

        for (int iter = 0; iter < cfg.num_iters; ++iter)
        {

            compute_gradient_block<<<blocks, threads, n_features * sizeof(float)>>>(
                d_X, d_y, d_w, d_block_grads,
                n_samples, n_features);

            reduce_block_grads<<<(n_features + threads - 1) / threads, threads>>>(
                d_block_grads,
                d_grad,
                blocks,
                n_features);

            update_weights<<<(n_features + threads - 1) / threads, threads>>>(
                d_w,
                d_grad,
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
        cudaFree(d_block_grads);
    }

} // namespace linear_regression