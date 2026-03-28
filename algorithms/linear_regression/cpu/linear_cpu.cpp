#include "../include/linear_regression.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace linear_regression
{

    void train_cpu(
        const float *X,
        const float *y,
        float *w,
        int n_samples,
        int n_features,
        const Config &cfg)
    {
        std::vector<float> grad(n_features);

        for (int iter = 0; iter < cfg.num_iters; ++iter)
        {

            // Zero gradient
            std::fill(grad.begin(), grad.end(), 0.0f);

            // Compute gradient of squared loss
            for (int i = 0; i < n_samples; ++i)
            {

                // Compute prediction: X_i · w
                float pred = 0.0f;
                const float *Xi = &X[i * n_features];

                for (int j = 0; j < n_features; ++j)
                {
                    pred += Xi[j] * w[j];
                }

                float residual = pred - y[i];

                // Accumulate gradient
                for (int j = 0; j < n_features; ++j)
                {
                    grad[j] += Xi[j] * residual;
                }
            }

            float scale = cfg.learning_rate / static_cast<float>(n_samples);

            // Apply regularization + update
            for (int j = 0; j < n_features; ++j)
            {

                float g = grad[j];

                // L2 regularization
                if (cfg.l2 != 0.0f)
                {
                    g += cfg.l2 * w[j];
                }

                // L1 regularization (subgradient)
                if (cfg.l1 != 0.0f)
                {
                    if (w[j] > 0.0f)
                        g += cfg.l1;
                    else if (w[j] < 0.0f)
                        g -= cfg.l1;
                    // if w[j] == 0, subgradient can be [-λ, λ]
                }

                w[j] -= scale * g;
            }
        }
    }

    float compute_mse(
        const float *X,
        const float *y,
        const float *w,
        int n_samples,
        int n_features)
    {
        float loss = 0.0f;

        for (int i = 0; i < n_samples; ++i)
        {

            float pred = 0.0f;
            const float *Xi = &X[i * n_features];

            for (int j = 0; j < n_features; ++j)
            {
                pred += Xi[j] * w[j];
            }

            float diff = pred - y[i];
            loss += diff * diff;
        }

        return loss / static_cast<float>(n_samples);
    }

} // namespace linear_regression