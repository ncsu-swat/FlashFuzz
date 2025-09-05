#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for configuration
        if (Size < 10) {
            return 0;
        }

        // Parse input tensor - batch norm expects at least 2D tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for batch norm
        if (input.dim() < 2) {
            // Reshape to add batch dimension if needed
            if (input.dim() == 0) {
                input = input.reshape({1, 1});
            } else if (input.dim() == 1) {
                input = input.reshape({1, input.size(0)});
            }
        }
        
        // Get number of features (channel dimension)
        int64_t num_features = input.size(1);
        
        // Parse configuration bytes if available
        bool use_weight = true;
        bool use_bias = true;
        bool training = false;
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset < Size) {
            uint8_t config_byte = Data[offset++];
            use_weight = (config_byte & 0x01) != 0;
            use_bias = (config_byte & 0x02) != 0;
            training = (config_byte & 0x04) != 0;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp momentum to reasonable range
            momentum = std::abs(momentum);
            if (!std::isfinite(momentum)) momentum = 0.1;
            momentum = std::fmod(momentum, 1.0); // Keep in [0, 1)
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive and reasonable
            eps = std::abs(eps);
            if (!std::isfinite(eps) || eps == 0) eps = 1e-5;
            eps = std::min(eps, 1.0); // Cap at 1.0
        }
        
        // Create weight and bias tensors if needed
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (use_weight && offset < Size) {
            try {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure weight matches num_features
                if (weight.numel() != num_features) {
                    weight = torch::randn({num_features}, input.options());
                }
                weight = weight.reshape({num_features});
            } catch (...) {
                weight = torch::ones({num_features}, input.options());
            }
        } else if (use_weight) {
            weight = torch::ones({num_features}, input.options());
        }
        
        if (use_bias && offset < Size) {
            try {
                bias = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure bias matches num_features
                if (bias.numel() != num_features) {
                    bias = torch::randn({num_features}, input.options());
                }
                bias = bias.reshape({num_features});
            } catch (...) {
                bias = torch::zeros({num_features}, input.options());
            }
        } else if (use_bias) {
            bias = torch::zeros({num_features}, input.options());
        }
        
        // Create running mean and variance
        torch::Tensor running_mean;
        torch::Tensor running_var;
        
        if (offset < Size) {
            try {
                running_mean = fuzzer_utils::createTensor(Data, Size, offset);
                if (running_mean.numel() != num_features) {
                    running_mean = torch::zeros({num_features}, input.options());
                }
                running_mean = running_mean.reshape({num_features});
            } catch (...) {
                running_mean = torch::zeros({num_features}, input.options());
            }
        } else {
            running_mean = torch::zeros({num_features}, input.options());
        }
        
        if (offset < Size) {
            try {
                running_var = fuzzer_utils::createTensor(Data, Size, offset);
                if (running_var.numel() != num_features) {
                    running_var = torch::ones({num_features}, input.options());
                }
                running_var = running_var.reshape({num_features});
                // Ensure variance is non-negative
                running_var = running_var.abs();
            } catch (...) {
                running_var = torch::ones({num_features}, input.options());
            }
        } else {
            running_var = torch::ones({num_features}, input.options());
        }
        
        // Convert tensors to appropriate dtype if needed (batch_norm typically expects float)
        if (input.dtype() != torch::kFloat32 && input.dtype() != torch::kFloat64 && 
            input.dtype() != torch::kFloat16 && input.dtype() != torch::kBFloat16) {
            input = input.to(torch::kFloat32);
            if (weight.defined()) weight = weight.to(torch::kFloat32);
            if (bias.defined()) bias = bias.to(torch::kFloat32);
            running_mean = running_mean.to(torch::kFloat32);
            running_var = running_var.to(torch::kFloat32);
        }
        
        // Test different batch norm variants
        try {
            // Standard batch_norm call
            torch::Tensor output = torch::batch_norm(
                input,
                weight.defined() ? weight : torch::Tensor(),
                bias.defined() ? bias : torch::Tensor(),
                running_mean,
                running_var,
                training,
                momentum,
                eps,
                false  // cudnn_enabled
            );
            
            // Additional edge case: zero-sized batch
            if (input.size(0) == 0) {
                torch::Tensor zero_batch = torch::empty({0, num_features}, input.options());
                torch::Tensor zero_output = torch::batch_norm(
                    zero_batch,
                    weight.defined() ? weight : torch::Tensor(),
                    bias.defined() ? bias : torch::Tensor(),
                    running_mean,
                    running_var,
                    false,  // training must be false for empty batch
                    momentum,
                    eps,
                    false
                );
            }
            
            // Test with different input shapes if we have 3D or 4D input
            if (input.dim() == 3) {
                // Test with permuted dimensions
                torch::Tensor permuted = input.permute({0, 2, 1});
                torch::Tensor perm_output = torch::batch_norm(
                    permuted,
                    weight.defined() ? weight : torch::Tensor(),
                    bias.defined() ? bias : torch::Tensor(),
                    running_mean,
                    running_var,
                    training,
                    momentum,
                    eps,
                    false
                );
            } else if (input.dim() == 4) {
                // Test with NCHW -> NHWC conversion
                torch::Tensor nhwc = input.permute({0, 2, 3, 1});
                // Note: This will fail as batch_norm expects channel dim at position 1
                // but we want to test error handling
                try {
                    torch::Tensor nhwc_output = torch::batch_norm(
                        nhwc,
                        weight.defined() ? weight : torch::Tensor(),
                        bias.defined() ? bias : torch::Tensor(),
                        running_mean,
                        running_var,
                        training,
                        momentum,
                        eps,
                        false
                    );
                } catch (...) {
                    // Expected to fail for mismatched dimensions
                }
            }
            
            // Test with non-contiguous tensor
            if (input.numel() > 1 && input.dim() > 1) {
                torch::Tensor non_contig = input.transpose(0, 1).transpose(0, 1);
                torch::Tensor nc_output = torch::batch_norm(
                    non_contig,
                    weight.defined() ? weight : torch::Tensor(),
                    bias.defined() ? bias : torch::Tensor(),
                    running_mean,
                    running_var,
                    training,
                    momentum,
                    eps,
                    false
                );
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid configurations
            // Continue fuzzing
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}