#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data to create a tensor
        if (Size < 3) {
            // Still try to create something minimal
            auto t = torch::zeros({1});
            torch::special::round(t);
            return 0;
        }

        // Create primary tensor from fuzzer input
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with a simple default tensor
            input_tensor = torch::randn({2, 2});
        }

        // Test basic round operation
        torch::Tensor result = torch::special::round(input_tensor);

        // Test with different tensor configurations if we have more data
        if (offset < Size) {
            // Try creating another tensor with remaining data
            try {
                torch::Tensor second_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                torch::special::round(second_tensor);
                
                // Test with views and non-contiguous tensors
                if (second_tensor.numel() > 1) {
                    auto transposed = second_tensor.t();
                    torch::special::round(transposed);
                    
                    auto sliced = second_tensor.narrow(0, 0, 1);
                    torch::special::round(sliced);
                }
            } catch (...) {
                // Continue with other tests
            }
        }

        // Test edge cases with special values if tensor is floating point
        if (input_tensor.is_floating_point()) {
            // Create tensors with special values
            auto options = input_tensor.options();
            
            // Test with infinity values
            auto inf_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity(), options);
            torch::special::round(inf_tensor);
            
            auto neg_inf_tensor = torch::full({2, 2}, -std::numeric_limits<float>::infinity(), options);
            torch::special::round(neg_inf_tensor);
            
            // Test with NaN values
            auto nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), options);
            torch::special::round(nan_tensor);
            
            // Test with very small values near zero
            auto small_tensor = torch::full({2, 2}, std::numeric_limits<float>::denorm_min(), options);
            torch::special::round(small_tensor);
            
            // Test with values at rounding boundaries
            auto boundary_tensor = torch::tensor({0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f}, options);
            torch::special::round(boundary_tensor);
        }

        // Test with different tensor shapes based on fuzzer input
        if (Size > 10) {
            uint8_t shape_selector = Data[Size % Size];
            
            // Empty tensor
            if (shape_selector % 7 == 0) {
                auto empty_tensor = torch::empty({0}, input_tensor.options());
                torch::special::round(empty_tensor);
            }
            
            // Scalar tensor
            if (shape_selector % 7 == 1) {
                auto scalar_tensor = torch::tensor(3.7, input_tensor.options());
                torch::special::round(scalar_tensor);
            }
            
            // High-dimensional tensor
            if (shape_selector % 7 == 2) {
                try {
                    auto high_dim = torch::randn({2, 1, 3, 1, 2}, input_tensor.options());
                    torch::special::round(high_dim);
                } catch (...) {
                    // Ignore allocation failures for large tensors
                }
            }
            
            // Single element tensor
            if (shape_selector % 7 == 3) {
                auto single = torch::ones({1}, input_tensor.options());
                torch::special::round(single);
            }
        }

        // Test in-place operation if supported
        try {
            auto clone = input_tensor.clone();
            clone = torch::special::round(clone);
        } catch (...) {
            // In-place might not be supported for all dtypes
        }

        // Test with requires_grad if floating point
        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            try {
                auto grad_tensor = input_tensor.clone().requires_grad_(true);
                auto rounded = torch::special::round(grad_tensor);
                // Note: round is not differentiable, but we test the operation anyway
            } catch (...) {
                // Gradient operations might fail
            }
        }

        // Test with different memory layouts
        if (input_tensor.dim() >= 2) {
            // Test with permuted tensor (non-contiguous)
            auto perm_dims = std::vector<int64_t>();
            for (int64_t i = input_tensor.dim() - 1; i >= 0; --i) {
                perm_dims.push_back(i);
            }
            auto permuted = input_tensor.permute(perm_dims);
            torch::special::round(permuted);
        }

        // Additional dtype-specific tests
        if (offset + 1 < Size) {
            uint8_t dtype_test = Data[offset++];
            
            // Test with different dtypes
            std::vector<torch::ScalarType> test_dtypes = {
                torch::kFloat32, torch::kFloat64, torch::kFloat16, 
                torch::kBFloat16, torch::kInt32, torch::kInt64
            };
            
            for (size_t i = 0; i < test_dtypes.size(); ++i) {
                if ((dtype_test >> i) & 1) {
                    try {
                        auto converted = input_tensor.to(test_dtypes[i]);
                        torch::special::round(converted);
                    } catch (...) {
                        // Type conversion might fail
                    }
                }
            }
        }

        // Test with complex numbers if applicable
        if (input_tensor.is_complex()) {
            // Complex rounding should work component-wise
            torch::special::round(input_tensor);
        }

        // Verify output properties
        if (result.defined()) {
            // Check that output has same shape as input
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Warning: Output shape mismatch" << std::endl;
            }
            
            // For floating point inputs, verify rounding behavior
            if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                // Check a few elements to ensure they are integers
                auto result_flat = result.flatten();
                auto input_flat = input_tensor.flatten();
                
                for (int64_t i = 0; i < std::min(int64_t(5), result_flat.numel()); ++i) {
                    float rounded_val = result_flat[i].item<float>();
                    float input_val = input_flat[i].item<float>();
                    
                    // Verify that rounded value is indeed an integer (within floating point precision)
                    if (!std::isnan(input_val) && !std::isinf(input_val)) {
                        float fractional_part = std::abs(rounded_val - std::round(rounded_val));
                        if (fractional_part > 1e-6) {
                            std::cerr << "Warning: Non-integer result from round" << std::endl;
                        }
                    }
                }
            }
        }

        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
}