#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for normalization modules
        uint8_t num_features = 0;
        if (offset < Size) {
            num_features = Data[offset++] % 64 + 1; // Ensure at least 1 feature
        }
        
        // Extract eps parameter
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and not too small
            if (std::isfinite(eps_raw) && eps_raw > 0) {
                eps = eps_raw;
            }
        }
        
        // Extract momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure momentum is between 0 and 1
            if (std::isfinite(momentum_raw) && momentum_raw >= 0 && momentum_raw <= 1) {
                momentum = momentum_raw;
            }
        }
        
        // Extract affine parameter
        bool affine = false;
        if (offset < Size) {
            affine = Data[offset++] % 2 == 0;
        }
        
        // Extract track_running_stats parameter
        bool track_running_stats = false;
        if (offset < Size) {
            track_running_stats = Data[offset++] % 2 == 0;
        }
        
        // 1. BatchNorm1d - expects (N, C) or (N, C, L)
        if (input.dim() == 2 || input.dim() == 3) {
            try {
                torch::nn::BatchNorm1dOptions bn1d_options(num_features);
                bn1d_options.eps(eps)
                           .momentum(momentum)
                           .affine(affine)
                           .track_running_stats(track_running_stats);
                
                auto bn1d = torch::nn::BatchNorm1d(bn1d_options);
                auto output = bn1d->forward(input);
            } catch (...) {
                // Silently catch exceptions from shape mismatch
            }
        }
        
        // 2. BatchNorm2d - expects (N, C, H, W)
        if (input.dim() == 4) {
            try {
                torch::nn::BatchNorm2dOptions bn2d_options(num_features);
                bn2d_options.eps(eps)
                           .momentum(momentum)
                           .affine(affine)
                           .track_running_stats(track_running_stats);
                
                auto bn2d = torch::nn::BatchNorm2d(bn2d_options);
                auto output = bn2d->forward(input);
            } catch (...) {
                // Silently catch exceptions from shape mismatch
            }
        }
        
        // 3. BatchNorm3d - expects (N, C, D, H, W)
        if (input.dim() == 5) {
            try {
                torch::nn::BatchNorm3dOptions bn3d_options(num_features);
                bn3d_options.eps(eps)
                           .momentum(momentum)
                           .affine(affine)
                           .track_running_stats(track_running_stats);
                
                auto bn3d = torch::nn::BatchNorm3d(bn3d_options);
                auto output = bn3d->forward(input);
            } catch (...) {
                // Silently catch exceptions from shape mismatch
            }
        }
        
        // 4. LayerNorm
        if (input.dim() > 0) {
            try {
                // Create normalized_shape based on the last dimensions of input
                std::vector<int64_t> normalized_shape;
                int64_t dim_to_use = std::min(static_cast<int64_t>(3), input.dim());
                
                for (int64_t i = input.dim() - dim_to_use; i < input.dim(); i++) {
                    if (i >= 0) {
                        normalized_shape.push_back(input.size(i));
                    }
                }
                
                if (!normalized_shape.empty()) {
                    torch::nn::LayerNormOptions ln_options(normalized_shape);
                    ln_options.eps(eps).elementwise_affine(affine);
                    
                    auto ln = torch::nn::LayerNorm(ln_options);
                    auto output = ln->forward(input);
                }
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 5. InstanceNorm1d - expects (N, C, L)
        if (input.dim() == 3) {
            try {
                torch::nn::InstanceNorm1dOptions in1d_options(num_features);
                in1d_options.eps(eps)
                           .momentum(momentum)
                           .affine(affine)
                           .track_running_stats(track_running_stats);
                
                auto in1d = torch::nn::InstanceNorm1d(in1d_options);
                auto output = in1d->forward(input);
            } catch (...) {
                // Silently catch exceptions from shape mismatch
            }
        }
        
        // 6. InstanceNorm2d - expects (N, C, H, W)
        if (input.dim() == 4) {
            try {
                torch::nn::InstanceNorm2dOptions in2d_options(num_features);
                in2d_options.eps(eps)
                           .momentum(momentum)
                           .affine(affine)
                           .track_running_stats(track_running_stats);
                
                auto in2d = torch::nn::InstanceNorm2d(in2d_options);
                auto output = in2d->forward(input);
            } catch (...) {
                // Silently catch exceptions from shape mismatch
            }
        }
        
        // 7. InstanceNorm3d - expects (N, C, D, H, W)
        if (input.dim() == 5) {
            try {
                torch::nn::InstanceNorm3dOptions in3d_options(num_features);
                in3d_options.eps(eps)
                           .momentum(momentum)
                           .affine(affine)
                           .track_running_stats(track_running_stats);
                
                auto in3d = torch::nn::InstanceNorm3d(in3d_options);
                auto output = in3d->forward(input);
            } catch (...) {
                // Silently catch exceptions from shape mismatch
            }
        }
        
        // 8. GroupNorm - expects (N, C, *) where C must be divisible by num_groups
        if (input.dim() >= 2) {
            try {
                // Extract num_groups parameter
                int64_t num_groups = 1;
                if (offset < Size) {
                    num_groups = (Data[offset++] % 16) + 1; // Between 1 and 16 groups
                }
                
                // num_channels must be divisible by num_groups
                int64_t num_channels = num_features;
                // Adjust to make divisible
                if (num_channels % num_groups != 0) {
                    num_channels = num_groups * ((num_channels / num_groups) + 1);
                }
                
                torch::nn::GroupNormOptions gn_options(num_groups, num_channels);
                gn_options.eps(eps).affine(affine);
                
                auto gn = torch::nn::GroupNorm(gn_options);
                auto output = gn->forward(input);
            } catch (...) {
                // Silently catch exceptions from shape mismatch
            }
        }
        
        // 9. LocalResponseNorm
        if (input.dim() >= 3) {
            try {
                // Extract size parameter
                int64_t size = 5;
                if (offset < Size) {
                    size = (Data[offset++] % 10) + 1; // Between 1 and 10
                }
                
                // Extract alpha, beta, k parameters
                double alpha = 1e-4;
                double beta = 0.75;
                double k = 1.0;
                
                if (offset + sizeof(float) <= Size) {
                    float alpha_raw;
                    std::memcpy(&alpha_raw, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (std::isfinite(alpha_raw) && alpha_raw > 0) {
                        alpha = alpha_raw;
                    }
                }
                
                if (offset + sizeof(float) <= Size) {
                    float beta_raw;
                    std::memcpy(&beta_raw, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (std::isfinite(beta_raw)) {
                        beta = beta_raw;
                    }
                }
                
                if (offset + sizeof(float) <= Size) {
                    float k_raw;
                    std::memcpy(&k_raw, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (std::isfinite(k_raw) && k_raw > 0) {
                        k = k_raw;
                    }
                }
                
                torch::nn::LocalResponseNormOptions lrn_options(size);
                lrn_options.alpha(alpha).beta(beta).k(k);
                
                auto lrn = torch::nn::LocalResponseNorm(lrn_options);
                auto output = lrn->forward(input);
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 10. CrossMapLRN2d - expects 4D input (N, C, H, W)
        if (input.dim() == 4) {
            try {
                int64_t size = 5;
                if (offset < Size) {
                    size = (Data[offset++] % 10) + 1;
                }
                
                double alpha = 1e-4;
                double beta = 0.75;
                double k = 1.0;
                
                torch::nn::CrossMapLRN2dOptions cmlrn_options(size);
                cmlrn_options.alpha(alpha).beta(beta).k(k);
                
                auto cmlrn = torch::nn::CrossMapLRN2d(cmlrn_options);
                auto output = cmlrn->forward(input);
            } catch (...) {
                // Silently catch exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}