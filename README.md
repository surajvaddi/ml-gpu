## Implementation Checklist

This project implements multiple machine learning algorithms with
CPU baselines, custom GPU kernels (forward + backward), and cuBLAS
reference implementations. Progress is tracked via Makefile targets.

A checklist item is considered **complete** when the corresponding
`make` target builds and runs successfully.

---

### 1. Linear Regression (MSE)

- [ ] CPU implementation (`make linear_cpu`)
- [ ] GPU-A: Naive CUDA forward + backward (`make linear_gpu_naive`)
- [ ] GPU-B: Memory-optimized CUDA (`make linear_gpu_opt`)
- [ ] GPU-C: Fused backward CUDA (`make linear_gpu_fused`)
- [ ] cuBLAS reference (`make linear_cublas`)
- [ ] Benchmark + correctness check (`make linear_bench`)
- [ ] Nsight profiling captured (`make linear_profile`)

---

### 2. Lasso Regression (L1)

- [ ] CPU implementation (`make lasso_cpu`)
- [ ] GPU-A: Naive CUDA gradient + shrinkage (`make lasso_gpu_naive`)
- [ ] GPU-B: Fused elementwise kernel (`make lasso_gpu_opt`)
- [ ] GPU-C: Reduction-aware backward (`make lasso_gpu_fused`)
- [ ] cuBLAS reference (`make lasso_cublas`)
- [ ] Benchmark + correctness check (`make lasso_bench`)
- [ ] Nsight profiling captured (`make lasso_profile`)

---

### 3. Support Vector Machine (Linear)

- [ ] CPU implementation (`make svm_cpu`)
- [ ] GPU-A: Naive margin + loss kernels (`make svm_gpu_naive`)
- [ ] GPU-B: Fused margin + gradient (`make svm_gpu_opt`)
- [ ] GPU-C: Reduction-optimized loss (`make svm_gpu_fused`)
- [ ] cuBLAS reference (`make svm_cublas`)
- [ ] Benchmark + correctness check (`make svm_bench`)
- [ ] Nsight profiling captured (`make svm_profile`)

---

### 4. Fully Connected ANN (MLP)

- [ ] CPU implementation (`make mlp_cpu`)
- [ ] GPU-A: Layer-separated CUDA (`make mlp_gpu_naive`)
- [ ] GPU-B: Activation-fused CUDA (`make mlp_gpu_opt`)
- [ ] GPU-C: Fully fused forward/backward (`make mlp_gpu_fused`)
- [ ] cuBLAS reference (`make mlp_cublas`)
- [ ] Benchmark + correctness check (`make mlp_bench`)
- [ ] Nsight profiling captured (`make mlp_profile`)

---

### 5. CNN (2D Convolution)

- [ ] CPU implementation (`make cnn_cpu`)
- [ ] GPU-A: Direct convolution (`make cnn_gpu_naive`)
- [ ] GPU-B: Shared-memory tiled convolution (`make cnn_gpu_opt`)
- [ ] GPU-C: im2col-based convolution (`make cnn_gpu_fused`)
- [ ] cuBLAS reference (`make cnn_cublas`)
- [ ] Benchmark + correctness check (`make cnn_bench`)
- [ ] Nsight profiling captured (`make cnn_profile`)

---

### 6. RNN (Vanilla / GRU-lite)

- [ ] CPU implementation (`make rnn_cpu`)
- [ ] GPU-A: Per-timestep kernels (`make rnn_gpu_naive`)
- [ ] GPU-B: Timestep fusion (`make rnn_gpu_opt`)
- [ ] GPU-C: Unrolled sequence kernel (`make rnn_gpu_fused`)
- [ ] cuBLAS reference (`make rnn_cublas`)
- [ ] Benchmark + correctness check (`make rnn_bench`)
- [ ] Nsight profiling captured (`make rnn_profile`)

---

### 7. Autoencoder

- [ ] CPU implementation (`make autoenc_cpu`)
- [ ] GPU-A: Separate encoder/decoder (`make autoenc_gpu_naive`)
- [ ] GPU-B: Loss-fused kernel (`make autoenc_gpu_opt`)
- [ ] GPU-C: Tied-weight backward fusion (`make autoenc_gpu_fused`)
- [ ] cuBLAS reference (`make autoenc_cublas`)
- [ ] Benchmark + correctness check (`make autoenc_bench`)
- [ ] Nsight profiling captured (`make autoenc_profile`)

---

### 8. K-Means

- [ ] CPU implementation (`make kmeans_cpu`)
- [ ] GPU-A: Atomic centroid updates (`make kmeans_gpu_naive`)
- [ ] GPU-B: Privatized centroids (`make kmeans_gpu_opt`)
- [ ] GPU-C: Streamed mini-batch K-Means (`make kmeans_gpu_fused`)
- [ ] cuBLAS distance computation (`make kmeans_cublas`)
- [ ] Benchmark + correctness check (`make kmeans_bench`)
- [ ] Nsight profiling captured (`make kmeans_profile`)

---

### 9. Transformer Attention

- [ ] CPU implementation (`make attention_cpu`)
- [ ] GPU-A: Naive attention (`make attention_gpu_naive`)
- [ ] GPU-B: Tiled attention (`make attention_gpu_opt`)
- [ ] GPU-C: FlashAttention-style fused kernel (`make attention_gpu_fused`)
- [ ] cuBLAS QK^T reference (`make attention_cublas`)
- [ ] Benchmark + correctness check (`make attention_bench`)
- [ ] Nsight profiling captured (`make attention_profile`)

---

### 10. GAN (MLP-based)

- [ ] CPU implementation (`make gan_cpu`)
- [ ] GPU-A: Sequential G/D training (`make gan_gpu_naive`)
- [ ] GPU-B: Fused discriminator loss (`make gan_gpu_opt`)
- [ ] GPU-C: Multi-stream training (`make gan_gpu_fused`)
- [ ] cuBLAS reference (`make gan_cublas`)
- [ ] Benchmark + correctness check (`make gan_bench`)
- [ ] Nsight profiling captured (`make gan_profile`)

---

### Global Infrastructure

- [ ] Unified tensor abstraction
- [ ] CPU/GPU numerical correctness checker
- [ ] Benchmark harness with CSV output
- [ ] Nsight Compute + Systems profiling scripts
- [ ] Roofline model plots
- [ ] Reproducible experiment scripts
