#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data to create a tensor
        if (Size < 3) {
            // Still try to create something minimal
            torch::Tensor t = torch::zeros({}, torch::kBool);
            torch::logical_not(t);
            return 0;
        }

        // Create primary input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with minimal tensor
            input_tensor = torch::zeros({1}, torch::kFloat32);
        }

        // Test basic logical_not
        torch::Tensor result = torch::logical_not(input_tensor);
        
        // Test with output tensor if we have more data
        if (offset < Size) {
            uint8_t out_option = Data[offset++];
            
            if (out_option % 3 == 0) {
                // Test with pre-allocated output tensor of same shape
                torch::Tensor out = torch::empty_like(input_tensor, torch::kBool);
                torch::logical_not(input_tensor, out);
                
                // Verify output was written to
                if (out.numel() > 0) {
                    auto sum = out.sum();
                }
            } else if (out_option % 3 == 1) {
                // Test with pre-allocated output tensor of different dtype (should be converted to bool)
                torch::Tensor out = torch::empty_like(input_tensor);
                torch::logical_not(input_tensor, out);
            } else {
                // Test with mismatched shape output (should resize or error)
                try {
                    torch::Tensor out = torch::empty({1}, torch::kBool);
                    torch::logical_not(input_tensor, out);
                } catch (const c10::Error& e) {
                    // Expected for shape mismatch in some cases
                }
            }
        }

        // Test edge cases based on remaining data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 8) {
                case 0: {
                    // Test with scalar tensor
                    torch::Tensor scalar = torch::tensor(3.14f);
                    torch::logical_not(scalar);
                    break;
                }
                case 1: {
                    // Test with empty tensor
                    torch::Tensor empty = torch::empty({0});
                    torch::logical_not(empty);
                    break;
                }
                case 2: {
                    // Test with NaN values
                    if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                        input_tensor[0] = std::numeric_limits<float>::quiet_NaN();
                        torch::logical_not(input_tensor);
                    }
                    break;
                }
                case 3: {
                    // Test with infinity values
                    if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                        input_tensor[0] = std::numeric_limits<float>::infinity();
                        torch::logical_not(input_tensor);
                        input_tensor[0] = -std::numeric_limits<float>::infinity();
                        torch::logical_not(input_tensor);
                    }
                    break;
                }
                case 4: {
                    // Test with complex tensors if available
                    if (input_tensor.dtype() == torch::kComplexFloat || 
                        input_tensor.dtype() == torch::kComplexDouble) {
                        torch::logical_not(input_tensor);
                    } else {
                        // Convert to complex and test
                        try {
                            auto complex_t = input_tensor.to(torch::kComplexFloat);
                            torch::logical_not(complex_t);
                        } catch (const c10::Error& e) {
                            // Some dtypes can't convert to complex
                        }
                    }
                    break;
                }
                case 5: {
                    // Test with non-contiguous tensor
                    if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
                        auto transposed = input_tensor.transpose(0, 1);
                        torch::logical_not(transposed);
                    }
                    break;
                }
                case 6: {
                    // Test with view/slice
                    if (input_tensor.numel() > 2) {
                        auto sliced = input_tensor.narrow(0, 0, 1);
                        torch::logical_not(sliced);
                    }
                    break;
                }
                case 7: {
                    // Test chained operations
                    auto r1 = torch::logical_not(input_tensor);
                    auto r2 = torch::logical_not(r1);  // Double negation
                    
                    // For boolean logic, double negation should equal original (when converted to bool)
                    if (input_tensor.dtype() == torch::kBool) {
                        auto diff = torch::any(r2 != input_tensor);
                    }
                    break;
                }
            }
        }

        // Test with different memory layouts if we have more data
        if (offset < Size && input_tensor.dim() >= 2) {
            uint8_t layout_option = Data[offset++];
            
            if (layout_option % 2 == 0) {
                // Test with channels_last memory format (for 4D tensors)
                if (input_tensor.dim() == 4) {
                    try {
                        auto cl_tensor = input_tensor.to(torch::MemoryFormat::ChannelsLast);
                        torch::logical_not(cl_tensor);
                    } catch (const c10::Error& e) {
                        // Some configurations might not support channels_last
                    }
                }
            }
        }

        // Test batch operations if tensor has multiple elements
        if (input_tensor.numel() > 1) {
            // Reshape and test
            try {
                auto reshaped = input_tensor.reshape({-1});
                torch::logical_not(reshaped);
            } catch (const c10::Error& e) {
                // Reshape might fail for some configurations
            }
        }

        // Additional stress test: large tensor operation
        if (offset < Size) {
            uint8_t stress_test = Data[offset++];
            if (stress_test % 10 == 0) {
                try {
                    // Create a larger tensor for stress testing
                    torch::Tensor large = torch::randn({100, 100}, input_tensor.options());
                    torch::logical_not(large);
                } catch (const std::bad_alloc& e) {
                    // Memory allocation might fail
                } catch (const c10::Error& e) {
                    // Other errors possible with large tensors
                }
            }
        }

        // Verify result properties
        if (result.defined()) {
            // Result should always be boolean type
            if (result.dtype() != torch::kBool) {
                std::cerr << "Warning: logical_not result is not boolean type" << std::endl;
            }
            
            // Result shape should match input shape
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Warning: shape mismatch between input and result" << std::endl;
            }
        }

        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;  // Continue fuzzing
    }
    catch (const std::bad_alloc &e)
    {
        // Memory allocation failures are expected with large tensors
        return 0;  // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard input for unexpected exceptions
    }
    
    return 0;
}