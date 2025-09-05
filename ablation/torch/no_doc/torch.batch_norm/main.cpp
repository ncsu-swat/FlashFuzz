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
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for configuration
        if (Size < 10) {
            return 0;
        }

        // Parse configuration bytes
        uint8_t config_byte = Data[offset++];
        bool training = config_byte & 0x01;
        bool use_weight = (config_byte >> 1) & 0x01;
        bool use_bias = (config_byte >> 2) & 0x01;
        bool use_running_mean = (config_byte >> 3) & 0x01;
        bool use_running_var = (config_byte >> 4) & 0x01;
        bool cudnn_enabled = (config_byte >> 5) & 0x01;
        
        // Parse momentum and epsilon with fuzzer-controlled values
        float momentum_raw = consumeValue<float>(Data, offset, Size, 0.1f);
        float epsilon_raw = consumeValue<float>(Data, offset, Size, 1e-5f);
        
        // Constrain momentum to [0, 1] and epsilon to positive values
        float momentum = std::abs(momentum_raw);
        if (momentum > 1.0f) momentum = 1.0f / (1.0f + momentum); // Map to (0, 1)
        float epsilon = std::abs(epsilon_raw);
        if (epsilon == 0.0f) epsilon = 1e-8f; // Avoid exact zero
        
        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create input tensor, try with minimal default
            input = torch::randn({2, 3, 4, 4});
        }
        
        // Batch norm requires at least 2D input
        if (input.dim() < 2) {
            // Reshape to add batch dimension
            input = input.unsqueeze(0);
            if (input.dim() < 2) {
                input = input.unsqueeze(0);
            }
        }
        
        // Get number of features (channel dimension for 2D+ inputs)
        int64_t num_features = (input.dim() >= 2) ? input.size(1) : 1;
        
        // Handle edge case of 0 features
        if (num_features <= 0) {
            num_features = 1;
            input = input.reshape({input.size(0), 1, -1});
        }
        
        // Create optional parameters based on config
        torch::Tensor weight, bias, running_mean, running_var;
        
        if (use_weight) {
            try {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure weight has correct shape
                if (weight.numel() != num_features) {
                    weight = weight.flatten().slice(0, 0, num_features);
                    if (weight.numel() < num_features) {
                        weight = torch::ones({num_features}, input.options());
                    }
                }
                weight = weight.reshape({num_features});
            } catch (...) {
                weight = torch::ones({num_features}, input.options());
            }
        }
        
        if (use_bias) {
            try {
                bias = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure bias has correct shape
                if (bias.numel() != num_features) {
                    bias = bias.flatten().slice(0, 0, num_features);
                    if (bias.numel() < num_features) {
                        bias = torch::zeros({num_features}, input.options());
                    }
                }
                bias = bias.reshape({num_features});
            } catch (...) {
                bias = torch::zeros({num_features}, input.options());
            }
        }
        
        if (use_running_mean) {
            try {
                running_mean = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure running_mean has correct shape
                if (running_mean.numel() != num_features) {
                    running_mean = running_mean.flatten().slice(0, 0, num_features);
                    if (running_mean.numel() < num_features) {
                        running_mean = torch::zeros({num_features}, input.options());
                    }
                }
                running_mean = running_mean.reshape({num_features});
            } catch (...) {
                running_mean = torch::zeros({num_features}, input.options());
            }
        }
        
        if (use_running_var) {
            try {
                running_var = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure running_var has correct shape and is positive
                if (running_var.numel() != num_features) {
                    running_var = running_var.flatten().slice(0, 0, num_features);
                    if (running_var.numel() < num_features) {
                        running_var = torch::ones({num_features}, input.options());
                    }
                }
                running_var = running_var.reshape({num_features}).abs();
                // Avoid exact zero variance
                running_var = running_var + epsilon;
            } catch (...) {
                running_var = torch::ones({num_features}, input.options());
            }
        }
        
        // Test different input shapes and configurations
        try {
            // Call batch_norm with various parameter combinations
            torch::Tensor output = torch::batch_norm(
                input,
                weight.defined() ? weight : torch::Tensor(),
                bias.defined() ? bias : torch::Tensor(),
                running_mean.defined() ? running_mean : torch::Tensor(),
                running_var.defined() ? running_var : torch::Tensor(),
                training,
                momentum,
                epsilon,
                cudnn_enabled
            );
            
            // Verify output shape matches input shape
            if (output.sizes() != input.sizes()) {
                std::cerr << "Output shape mismatch: " << output.sizes() << " vs " << input.sizes() << std::endl;
            }
            
            // Test with different memory formats if applicable
            if (input.dim() == 4 && offset < Size) {
                uint8_t format_byte = Data[offset++];
                if (format_byte & 0x01) {
                    // Try channels_last format for 4D tensors
                    auto input_cl = input.to(torch::MemoryFormat::ChannelsLast);
                    torch::Tensor output_cl = torch::batch_norm(
                        input_cl,
                        weight.defined() ? weight : torch::Tensor(),
                        bias.defined() ? bias : torch::Tensor(),
                        running_mean.defined() ? running_mean : torch::Tensor(),
                        running_var.defined() ? running_var : torch::Tensor(),
                        training,
                        momentum,
                        epsilon,
                        cudnn_enabled
                    );
                }
            }
            
            // Test edge cases with special values
            if (offset < Size && (Data[offset++] & 0x01)) {
                // Test with NaN/Inf in input
                auto special_input = input.clone();
                if (special_input.numel() > 0) {
                    special_input.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                    if (special_input.numel() > 1) {
                        special_input.view(-1)[1] = std::numeric_limits<float>::infinity();
                    }
                    
                    torch::Tensor special_output = torch::batch_norm(
                        special_input,
                        weight.defined() ? weight : torch::Tensor(),
                        bias.defined() ? bias : torch::Tensor(),
                        running_mean.defined() ? running_mean : torch::Tensor(),
                        running_var.defined() ? running_var : torch::Tensor(),
                        training,
                        momentum,
                        epsilon,
                        cudnn_enabled
                    );
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid configurations
            // Continue fuzzing
        } catch (const std::exception& e) {
            // Log but continue for other exceptions
            std::cerr << "batch_norm exception: " << e.what() << std::endl;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}