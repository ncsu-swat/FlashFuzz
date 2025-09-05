#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 4 bytes: 2 for input tensor metadata, 2 for weight tensor metadata
        if (Size < 4) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we've consumed all data, create a simple weight
        if (offset >= Size) {
            // Default to scalar weight of 0.25
            torch::Tensor weight = torch::tensor(0.25f, input.options());
            torch::Tensor result = torch::prelu(input, weight);
            return 0;
        }

        // Parse weight configuration byte
        uint8_t weight_config = (offset < Size) ? Data[offset++] : 0;
        
        // Determine weight shape based on configuration and input
        torch::Tensor weight;
        
        if (weight_config % 3 == 0) {
            // Case 1: Scalar weight (0-D tensor)
            if (offset < Size) {
                float weight_val = static_cast<float>(Data[offset++]) / 255.0f;
                weight = torch::tensor(weight_val, input.options());
            } else {
                weight = torch::tensor(0.25f, input.options());
            }
        } else if (weight_config % 3 == 1) {
            // Case 2: 1-D weight matching number of channels
            int64_t num_channels = 1;
            if (input.dim() >= 2) {
                num_channels = input.size(1);
            }
            
            // Create 1-D weight tensor with appropriate size
            std::vector<int64_t> weight_shape = {num_channels};
            
            // Parse weight values or use random
            if (offset + num_channels <= Size) {
                std::vector<float> weight_data;
                weight_data.reserve(num_channels);
                for (int64_t i = 0; i < num_channels; ++i) {
                    if (offset < Size) {
                        weight_data.push_back(static_cast<float>(Data[offset++]) / 255.0f - 0.5f);
                    } else {
                        weight_data.push_back(0.25f);
                    }
                }
                weight = torch::from_blob(weight_data.data(), weight_shape, torch::kFloat).clone().to(input.options());
            } else {
                // Use random weight values
                weight = torch::randn(weight_shape, input.options()) * 0.5f;
            }
        } else {
            // Case 3: Try to create weight tensor from remaining data
            if (offset < Size) {
                try {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure weight is either scalar or 1-D
                    if (weight.dim() > 1) {
                        // Flatten to 1-D
                        weight = weight.flatten();
                        
                        // Adjust size if needed to match channels
                        if (input.dim() >= 2) {
                            int64_t num_channels = input.size(1);
                            if (weight.numel() != num_channels) {
                                // Resize weight to match channels
                                if (weight.numel() > num_channels) {
                                    weight = weight.slice(0, 0, num_channels);
                                } else if (weight.numel() > 0) {
                                    // Repeat to fill channels
                                    int64_t repeat_times = (num_channels + weight.numel() - 1) / weight.numel();
                                    weight = weight.repeat({repeat_times}).slice(0, 0, num_channels);
                                }
                            }
                        } else {
                            // For input.dim() < 2, weight should be size 1
                            if (weight.numel() != 1) {
                                weight = weight[0].unsqueeze(0);
                            }
                        }
                    }
                } catch (...) {
                    // Fallback to scalar weight
                    weight = torch::tensor(0.25f, input.options());
                }
            } else {
                weight = torch::tensor(0.25f, input.options());
            }
        }

        // Convert weight to same dtype as input if different
        if (weight.dtype() != input.dtype()) {
            weight = weight.to(input.dtype());
        }

        // Apply PReLU operation
        torch::Tensor result = torch::prelu(input, weight);
        
        // Additional operations to increase coverage
        if (offset < Size && Data[offset] % 4 == 0) {
            // Test in-place operation if possible
            torch::Tensor input_copy = input.clone();
            input_copy = torch::prelu(input_copy, weight);
        }
        
        // Test with different memory layouts
        if (offset < Size && Data[offset] % 3 == 0 && input.dim() > 1) {
            // Test with non-contiguous tensor
            torch::Tensor transposed = input.transpose(0, -1);
            torch::Tensor result_transposed = torch::prelu(transposed, weight);
        }
        
        // Test gradient computation if dtype supports it
        if ((input.dtype() == torch::kFloat || input.dtype() == torch::kDouble || 
             input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16) &&
            input.numel() > 0 && weight.numel() > 0) {
            
            try {
                torch::Tensor input_grad = input.clone().requires_grad_(true);
                torch::Tensor weight_grad = weight.clone().requires_grad_(true);
                
                torch::Tensor output = torch::prelu(input_grad, weight_grad);
                
                if (output.numel() > 0) {
                    // Compute gradients
                    torch::Tensor grad_output = torch::ones_like(output);
                    output.backward(grad_output);
                }
            } catch (...) {
                // Gradient computation might fail for some configurations
            }
        }
        
        // Test edge cases based on remaining fuzzer bytes
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 5 == 0 && input.numel() > 0) {
                // Test with zero weight
                torch::Tensor zero_weight = torch::zeros_like(weight);
                torch::Tensor zero_result = torch::prelu(input, zero_weight);
            } else if (edge_case % 5 == 1) {
                // Test with negative weight
                torch::Tensor neg_weight = -torch::abs(weight);
                torch::Tensor neg_result = torch::prelu(input, neg_weight);
            } else if (edge_case % 5 == 2) {
                // Test with large weight values
                torch::Tensor large_weight = weight * 100.0f;
                torch::Tensor large_result = torch::prelu(input, large_weight);
            } else if (edge_case % 5 == 3) {
                // Test with very small weight values
                torch::Tensor small_weight = weight * 0.001f;
                torch::Tensor small_result = torch::prelu(input, small_weight);
            }
        }

        return 0;
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
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
}