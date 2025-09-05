#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume a value from fuzzer data
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    
    // Clamp to range
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need minimum bytes for parameters and tensor creation
        if (Size < 10) {
            return 0;
        }

        // Parse LocalResponseNorm parameters from fuzzer input
        // Size parameter (must be odd, range 1-99)
        uint8_t size_byte = Data[offset++];
        int64_t norm_size = (size_byte % 50) * 2 + 1; // Ensures odd number between 1-99
        
        // Alpha parameter (multiplicative factor)
        float alpha = consumeValue<float>(Data, offset, Size, 1e-10f, 1.0f);
        
        // Beta parameter (exponent)
        float beta = consumeValue<float>(Data, offset, Size, 0.01f, 5.0f);
        
        // K parameter (additive factor)
        float k = consumeValue<float>(Data, offset, Size, 0.0f, 10.0f);
        
        // Create LocalResponseNorm module with fuzzed parameters
        torch::nn::LocalResponseNormOptions options(norm_size);
        options.alpha(alpha).beta(beta).k(k);
        torch::nn::LocalResponseNorm lrn_module(options);
        
        // Create input tensor from remaining fuzzer data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with a default tensor
            input = torch::randn({2, 3, 4, 4});
        }
        
        // Ensure tensor has appropriate dimensions for LocalResponseNorm
        // LRN expects at least 3D tensor (batch, channels, spatial dims...)
        if (input.dim() < 2) {
            // Reshape to add dimensions
            auto numel = input.numel();
            if (numel == 0) {
                input = torch::randn({1, 1, 1});
            } else if (numel == 1) {
                input = input.view({1, 1, 1});
            } else {
                // Try to create a reasonable shape
                int64_t channels = std::min(numel, int64_t(8));
                int64_t remaining = numel / channels;
                if (remaining == 0) remaining = 1;
                input = input.view({1, channels, remaining});
            }
        } else if (input.dim() == 2) {
            // Add batch dimension
            input = input.unsqueeze(0);
        }
        
        // Test different input types and configurations
        std::vector<torch::Tensor> test_tensors;
        test_tensors.push_back(input);
        
        // Also test with different memory layouts if tensor is large enough
        if (input.numel() > 1 && input.dim() >= 3) {
            // Test with non-contiguous tensor
            if (input.size(0) > 1 && input.size(1) > 1) {
                test_tensors.push_back(input.transpose(0, 1));
            }
            
            // Test with different dtype conversions
            if (input.dtype() != torch::kFloat32) {
                test_tensors.push_back(input.to(torch::kFloat32));
            }
            if (input.dtype() != torch::kFloat64) {
                test_tensors.push_back(input.to(torch::kFloat64));
            }
        }
        
        // Apply LocalResponseNorm to all test tensors
        for (auto& tensor : test_tensors) {
            try {
                // Skip if tensor doesn't have proper shape
                if (tensor.dim() < 3 || tensor.size(1) < 1) {
                    continue;
                }
                
                torch::Tensor output = lrn_module->forward(tensor);
                
                // Perform some basic checks on output
                if (output.numel() > 0) {
                    // Check for NaN/Inf
                    bool has_nan = torch::isnan(output).any().item<bool>();
                    bool has_inf = torch::isinf(output).any().item<bool>();
                    
                    if (has_nan || has_inf) {
                        // This is interesting but not necessarily a bug
                        continue;
                    }
                    
                    // Verify output shape matches input
                    if (output.sizes() != tensor.sizes()) {
                        std::cerr << "Shape mismatch: input " << tensor.sizes() 
                                 << " vs output " << output.sizes() << std::endl;
                    }
                }
                
                // Test backward pass if tensor requires grad
                if (tensor.requires_grad() && output.numel() > 0) {
                    try {
                        auto grad_output = torch::ones_like(output);
                        output.backward(grad_output);
                    } catch (const std::exception& e) {
                        // Backward pass failed, but continue
                    }
                }
                
            } catch (const c10::Error& e) {
                // PyTorch-specific errors - these are expected for invalid inputs
                continue;
            } catch (const std::exception& e) {
                // Other exceptions might indicate bugs
                std::cerr << "Unexpected exception in forward pass: " << e.what() << std::endl;
                continue;
            }
        }
        
        // Additional edge case testing with extreme parameters
        if (Size % 7 == 0) {  // Randomly test extreme parameters
            try {
                torch::nn::LocalResponseNormOptions extreme_options(1);
                extreme_options.alpha(1e-8).beta(0.01).k(0.0);
                torch::nn::LocalResponseNorm extreme_lrn(extreme_options);
                
                auto small_input = torch::randn({1, 2, 3});
                extreme_lrn->forward(small_input);
            } catch (...) {
                // Ignore failures with extreme parameters
            }
        }
        
        if (Size % 11 == 0) {  // Test with large size parameter
            try {
                int64_t large_size = (Size % 100) * 2 + 1;
                if (large_size > 0 && large_size < 1000) {
                    torch::nn::LocalResponseNormOptions large_options(large_size);
                    torch::nn::LocalResponseNorm large_lrn(large_options);
                    
                    auto test_input = torch::randn({1, large_size + 1, 4});
                    large_lrn->forward(test_input);
                }
            } catch (...) {
                // Ignore failures
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}