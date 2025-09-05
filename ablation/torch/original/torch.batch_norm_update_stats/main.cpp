#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume a value from fuzzer data
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T default_val) {
    if (offset + sizeof(T) > size) {
        return default_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    if (Size < 10) {  // Need minimum bytes for basic parsing
        return 0;
    }

    try
    {
        size_t offset = 0;

        // Create input tensor - can be various shapes
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for batch norm
        if (input.dim() < 2) {
            // Reshape to have at least 2 dims
            auto numel = input.numel();
            if (numel > 0) {
                int64_t batch_size = std::max(int64_t(1), numel / 2);
                int64_t channels = std::max(int64_t(1), numel / batch_size);
                input = input.reshape({batch_size, channels});
            } else {
                input = input.reshape({0, 1});
            }
        }

        // Get number of channels (dimension 1)
        int64_t num_channels = input.size(1);
        
        // Parse momentum value
        double momentum = 0.1;  // default
        if (offset < Size) {
            uint8_t momentum_byte = Data[offset++];
            momentum = static_cast<double>(momentum_byte) / 255.0;  // normalize to [0, 1]
        }

        // Parse epsilon value for numerical stability
        double eps = 1e-5;  // default
        if (offset < Size) {
            uint8_t eps_byte = Data[offset++];
            eps = 1e-8 + (static_cast<double>(eps_byte) / 255.0) * 1e-3;  // range [1e-8, ~1e-3]
        }

        // Decide if we should use existing running stats or nullptr
        bool use_running_stats = false;
        if (offset < Size) {
            use_running_stats = (Data[offset++] % 2) == 0;
        }

        torch::Tensor running_mean, running_var;
        
        if (use_running_stats && num_channels > 0) {
            // Create running mean and variance tensors
            // Use remaining fuzzer data or create random
            if (offset + 2 < Size) {
                // Try to parse from fuzzer data
                try {
                    running_mean = fuzzer_utils::createTensor(Data, Size, offset);
                    // Ensure it's 1D with correct size
                    if (running_mean.numel() != num_channels) {
                        running_mean = running_mean.flatten().slice(0, 0, num_channels);
                        if (running_mean.numel() < num_channels) {
                            // Pad with zeros if needed
                            running_mean = torch::cat({running_mean, 
                                torch::zeros({num_channels - running_mean.numel()}, running_mean.options())});
                        }
                    }
                    running_mean = running_mean.to(input.dtype());
                } catch (...) {
                    running_mean = torch::zeros({num_channels}, input.options());
                }

                try {
                    running_var = fuzzer_utils::createTensor(Data, Size, offset);
                    // Ensure it's 1D with correct size
                    if (running_var.numel() != num_channels) {
                        running_var = running_var.flatten().slice(0, 0, num_channels);
                        if (running_var.numel() < num_channels) {
                            // Pad with ones if needed
                            running_var = torch::cat({running_var,
                                torch::ones({num_channels - running_var.numel()}, running_var.options())});
                        }
                    }
                    // Ensure variance is positive
                    running_var = running_var.abs() + eps;
                    running_var = running_var.to(input.dtype());
                } catch (...) {
                    running_var = torch::ones({num_channels}, input.options());
                }
            } else {
                // Create default running stats
                running_mean = torch::zeros({num_channels}, input.options());
                running_var = torch::ones({num_channels}, input.options());
            }
        }

        // Convert input to float if needed (batch norm typically works with float)
        if (input.dtype() != torch::kFloat && input.dtype() != torch::kDouble && 
            input.dtype() != torch::kHalf && input.dtype() != torch::kBFloat16) {
            input = input.to(torch::kFloat);
            if (use_running_stats) {
                running_mean = running_mean.to(torch::kFloat);
                running_var = running_var.to(torch::kFloat);
            }
        }

        // Call batch_norm_update_stats
        // This function computes batch statistics and optionally updates running stats
        torch::Tensor save_mean, save_invstd;
        
        if (use_running_stats && num_channels > 0) {
            // With running stats
            std::tie(save_mean, save_invstd) = torch::batch_norm_update_stats(
                input, running_mean, running_var, momentum);
            
            // Verify outputs have expected shape
            if (save_mean.numel() != num_channels || save_invstd.numel() != num_channels) {
                std::cerr << "Unexpected output shape from batch_norm_update_stats" << std::endl;
            }
        } else {
            // Without running stats (compute batch stats only)
            std::tie(save_mean, save_invstd) = torch::batch_norm_update_stats(
                input, {}, {}, momentum);
        }

        // Additional operations to increase coverage
        if (save_mean.numel() > 0 && save_invstd.numel() > 0) {
            // Test with different input shapes while keeping same channel count
            if (offset < Size && (Data[offset++] % 3) == 0) {
                // Try different spatial dimensions
                auto new_shape = input.sizes().vec();
                if (new_shape.size() >= 2) {
                    // Randomly modify spatial dimensions
                    for (size_t i = 2; i < new_shape.size(); ++i) {
                        if (offset < Size) {
                            new_shape[i] = 1 + (Data[offset++] % 8);
                        }
                    }
                    try {
                        auto reshaped_input = input.reshape(new_shape);
                        auto [mean2, invstd2] = torch::batch_norm_update_stats(
                            reshaped_input, running_mean, running_var, momentum);
                    } catch (...) {
                        // Reshape might fail, that's ok
                    }
                }
            }

            // Test with transposed/permuted input
            if (input.dim() >= 3 && offset < Size && (Data[offset++] % 2) == 0) {
                std::vector<int64_t> perm;
                for (int64_t i = 0; i < input.dim(); ++i) {
                    perm.push_back(i);
                }
                // Keep batch and channel dims, shuffle others
                if (perm.size() > 2) {
                    std::swap(perm[2], perm[perm.size() - 1]);
                }
                try {
                    auto permuted = input.permute(perm);
                    auto [mean3, invstd3] = torch::batch_norm_update_stats(
                        permuted, running_mean, running_var, momentum);
                } catch (...) {
                    // Permute might fail for some configurations
                }
            }
        }

        // Test edge cases
        if (offset < Size && (Data[offset++] % 4) == 0) {
            // Test with zero momentum
            try {
                auto [mean_zero, invstd_zero] = torch::batch_norm_update_stats(
                    input, running_mean, running_var, 0.0);
            } catch (...) {}
            
            // Test with momentum = 1.0
            try {
                auto [mean_one, invstd_one] = torch::batch_norm_update_stats(
                    input, running_mean, running_var, 1.0);
            } catch (...) {}
        }

        // Test with different memory layouts
        if (!input.is_contiguous() || (offset < Size && (Data[offset++] % 2) == 0)) {
            try {
                auto non_contig = input.transpose(0, input.dim() - 1);
                auto [mean_nc, invstd_nc] = torch::batch_norm_update_stats(
                    non_contig, {}, {}, momentum);
            } catch (...) {}
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Catch any other exceptions
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}