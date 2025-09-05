#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 10) {
            // Still try to create minimal tensors
            auto input = torch::randn({1});
            auto weight = torch::randn({1});
            auto result = torch::prelu(input, weight);
            return 0;
        }

        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create input, try with random small tensor
            input = torch::randn({2, 3});
        }

        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            try {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // Default to scalar weight
                weight = torch::randn({1});
            }
        } else {
            // Use remaining bytes to determine weight characteristics
            if (Size > 0) {
                uint8_t weight_type = Data[Size - 1] % 4;
                switch (weight_type) {
                    case 0:
                        weight = torch::randn({1}); // Scalar
                        break;
                    case 1:
                        // Try to match input channels if possible
                        if (input.dim() >= 2) {
                            weight = torch::randn({input.size(1)});
                        } else {
                            weight = torch::randn({1});
                        }
                        break;
                    case 2:
                        weight = torch::zeros({1}); // Edge case: zero weight
                        break;
                    case 3:
                        weight = torch::ones({1}) * -1; // Negative weight
                        break;
                }
            } else {
                weight = torch::randn({1});
            }
        }

        // Additional tensor modifications based on remaining bytes
        if (offset < Size && (Size - offset) > 0) {
            uint8_t mod_byte = Data[offset++];
            
            // Modify input tensor properties
            if (mod_byte & 0x01) {
                // Make input require gradient
                if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble ||
                    input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16) {
                    input = input.requires_grad_(true);
                }
            }
            
            if (mod_byte & 0x02) {
                // Make weight require gradient
                if (weight.dtype() == torch::kFloat || weight.dtype() == torch::kDouble ||
                    weight.dtype() == torch::kHalf || weight.dtype() == torch::kBFloat16) {
                    weight = weight.requires_grad_(true);
                }
            }
            
            if (mod_byte & 0x04) {
                // Try to make tensors non-contiguous
                if (input.numel() > 1 && input.dim() > 1) {
                    input = input.transpose(0, -1);
                }
            }
            
            if (mod_byte & 0x08) {
                // Add extreme values
                if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                    input[0] = std::numeric_limits<float>::infinity();
                }
            }
            
            if (mod_byte & 0x10) {
                // Add NaN values
                if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                    if (input.numel() > 1) {
                        input[1] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
        }

        // Test various weight shapes for broadcasting
        if (offset < Size && (Size - offset) > 0) {
            uint8_t shape_selector = Data[offset++] % 5;
            try {
                switch (shape_selector) {
                    case 0:
                        // Scalar weight (already default)
                        break;
                    case 1:
                        // Weight matching number of channels (if applicable)
                        if (input.dim() >= 2 && input.size(0) > 0) {
                            int64_t num_channels = input.size(1);
                            if (num_channels > 0 && num_channels < 10000) {
                                weight = torch::randn({num_channels});
                            }
                        }
                        break;
                    case 2:
                        // Empty weight tensor
                        weight = torch::empty({0});
                        break;
                    case 3:
                        // Large weight tensor
                        weight = torch::randn({100});
                        break;
                    case 4:
                        // Multi-dimensional weight (should trigger error or broadcasting)
                        weight = torch::randn({2, 3});
                        break;
                }
            } catch (...) {
                // Keep original weight if reshape fails
            }
        }

        // Perform PReLU operation
        torch::Tensor result;
        try {
            result = torch::prelu(input, weight);
            
            // Verify output properties
            if (result.defined()) {
                // Check shape matches input
                if (result.sizes() != input.sizes()) {
                    std::cerr << "Warning: Output shape mismatch" << std::endl;
                }
                
                // Check for NaN/Inf in output
                if (result.dtype() == torch::kFloat || result.dtype() == torch::kDouble) {
                    bool has_nan = torch::any(torch::isnan(result)).item<bool>();
                    bool has_inf = torch::any(torch::isinf(result)).item<bool>();
                    
                    if (has_nan && !torch::any(torch::isnan(input)).item<bool>() && 
                        !torch::any(torch::isnan(weight)).item<bool>()) {
                        std::cerr << "Warning: NaN introduced by PReLU" << std::endl;
                    }
                }
                
                // Test backward pass if gradients are enabled
                if (input.requires_grad() && result.requires_grad()) {
                    try {
                        auto grad_output = torch::ones_like(result);
                        result.backward(grad_output);
                    } catch (const std::exception& e) {
                        // Backward pass failed, but forward pass succeeded
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors (like shape mismatches) are expected
            return 0;
        }

        // Additional operations to increase coverage
        if (result.defined() && offset < Size) {
            uint8_t extra_ops = Data[offset++];
            
            if (extra_ops & 0x01) {
                // Try in-place operation
                try {
                    auto input_copy = input.clone();
                    torch::prelu_(input_copy, weight);
                } catch (...) {
                    // In-place operation might fail for certain dtypes
                }
            }
            
            if (extra_ops & 0x02) {
                // Try with different memory formats
                if (input.dim() == 4) {
                    try {
                        auto channels_last_input = input.to(torch::MemoryFormat::ChannelsLast);
                        auto result2 = torch::prelu(channels_last_input, weight);
                    } catch (...) {
                        // Memory format conversion might fail
                    }
                }
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}