#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }

        // Parse negative slope parameter for LeakyReLU
        double negative_slope = 0.01; // default value
        if (offset + sizeof(float) <= Size) {
            float slope_raw;
            std::memcpy(&slope_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Bound the slope to reasonable range to avoid numerical issues
            // but still allow diverse values including edge cases
            if (!std::isnan(slope_raw) && !std::isinf(slope_raw)) {
                negative_slope = static_cast<double>(slope_raw);
                // Allow wide range including negative values for thorough testing
                negative_slope = std::fmod(negative_slope, 10.0);
            }
        }

        // Parse inplace flag
        bool inplace = false;
        if (offset < Size) {
            inplace = (Data[offset++] % 2) == 1;
        }

        // Create LeakyReLU module with parsed parameters
        auto leaky_relu_options = torch::nn::LeakyReLUOptions()
            .negative_slope(negative_slope)
            .inplace(inplace);
        
        torch::nn::LeakyReLU leaky_relu(leaky_relu_options);

        // Parse number of tensors to test (1-5)
        uint8_t num_tensors = 1;
        if (offset < Size) {
            num_tensors = (Data[offset++] % 5) + 1;
        }

        // Process multiple tensors to increase coverage
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                // Create input tensor from fuzzer data
                torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test with different tensor properties
                if (offset < Size && (Data[offset++] % 2) == 1) {
                    // Make tensor non-contiguous sometimes
                    if (input.dim() > 1 && input.size(0) > 1 && input.size(1) > 1) {
                        input = input.transpose(0, 1);
                    }
                }
                
                // Test with requires_grad sometimes
                if (offset < Size && (Data[offset++] % 2) == 1) {
                    // Only set requires_grad for floating point types
                    if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble || 
                        input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16) {
                        input.requires_grad_(true);
                    }
                }

                // Apply LeakyReLU
                torch::Tensor output = leaky_relu->forward(input);
                
                // Perform additional operations to exercise more code paths
                if (output.requires_grad() && offset < Size && (Data[offset++] % 2) == 1) {
                    // Test backward pass
                    torch::Tensor grad_output = torch::ones_like(output);
                    output.backward(grad_output);
                }
                
                // Test some properties of the output
                if (offset < Size) {
                    uint8_t op = Data[offset++] % 6;
                    switch(op) {
                        case 0:
                            // Check if output has same shape as input
                            if (output.sizes() != input.sizes() && !inplace) {
                                std::cerr << "Shape mismatch after LeakyReLU" << std::endl;
                            }
                            break;
                        case 1:
                            // Sum reduction
                            if (output.numel() > 0) {
                                auto sum_val = output.sum();
                            }
                            break;
                        case 2:
                            // Mean reduction
                            if (output.numel() > 0 && output.dtype() != torch::kBool) {
                                auto mean_val = output.mean();
                            }
                            break;
                        case 3:
                            // Max reduction
                            if (output.numel() > 0) {
                                auto max_val = output.max();
                            }
                            break;
                        case 4:
                            // Min reduction  
                            if (output.numel() > 0) {
                                auto min_val = output.min();
                            }
                            break;
                        case 5:
                            // Check contiguity
                            bool is_contiguous = output.is_contiguous();
                            break;
                    }
                }
                
                // Test with different device types if available
                if (torch::cuda::is_available() && offset < Size && (Data[offset++] % 4) == 0) {
                    try {
                        auto cuda_input = input.to(torch::kCUDA);
                        auto cuda_output = leaky_relu->forward(cuda_input);
                        // Move back to CPU for potential comparison
                        cuda_output = cuda_output.to(torch::kCPU);
                    } catch (const c10::Error& e) {
                        // CUDA operations might fail, continue testing
                    }
                }
                
            } catch (const c10::Error& e) {
                // PyTorch errors for this specific tensor, continue with next
                continue;
            } catch (const std::runtime_error& e) {
                // Tensor creation errors, continue with next
                continue;
            }
        }
        
        // Test edge cases with manually created tensors
        if (Size > 0) {
            uint8_t edge_case = Data[0] % 8;
            try {
                torch::Tensor edge_input;
                switch(edge_case) {
                    case 0:
                        // Empty tensor
                        edge_input = torch::empty({0});
                        break;
                    case 1:
                        // Scalar tensor
                        edge_input = torch::tensor(3.14f);
                        break;
                    case 2:
                        // Large tensor
                        edge_input = torch::randn({100, 100});
                        break;
                    case 3:
                        // Tensor with inf values
                        edge_input = torch::tensor({std::numeric_limits<float>::infinity(), 
                                                   -std::numeric_limits<float>::infinity(), 
                                                   0.0f});
                        break;
                    case 4:
                        // Tensor with nan values
                        edge_input = torch::tensor({std::numeric_limits<float>::quiet_NaN(), 
                                                   1.0f, -1.0f});
                        break;
                    case 5:
                        // Very small values
                        edge_input = torch::tensor({std::numeric_limits<float>::min(), 
                                                   std::numeric_limits<float>::epsilon(), 
                                                   -std::numeric_limits<float>::min()});
                        break;
                    case 6:
                        // Mixed positive and negative
                        edge_input = torch::tensor({-5.0f, -1.0f, 0.0f, 1.0f, 5.0f});
                        break;
                    case 7:
                        // High dimensional tensor
                        edge_input = torch::randn({2, 3, 4, 5});
                        break;
                }
                
                auto edge_output = leaky_relu->forward(edge_input);
                
            } catch (const c10::Error& e) {
                // Edge case failed, acceptable
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