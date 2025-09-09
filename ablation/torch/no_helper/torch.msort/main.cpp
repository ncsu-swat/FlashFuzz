#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and dtypes
        auto input_tensor = generateTensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Test basic msort functionality
        auto result1 = torch::msort(input_tensor);

        // Test with output tensor
        auto out_tensor = torch::empty_like(input_tensor);
        auto result2 = torch::msort(input_tensor, out_tensor);

        // Verify that result2 and out_tensor are the same
        if (!torch::allclose(result2, out_tensor, 1e-5, 1e-8, /*equal_nan=*/true)) {
            std::cerr << "msort with out parameter failed consistency check" << std::endl;
        }

        // Test edge cases with different tensor shapes
        if (input_tensor.dim() > 0) {
            // Test with 1D tensor (reshape to 1D)
            auto flat_tensor = input_tensor.flatten();
            auto result_flat = torch::msort(flat_tensor);
            
            // Test with reshaped tensor
            if (input_tensor.numel() >= 4) {
                auto reshaped = input_tensor.view({-1, 2});
                auto result_reshaped = torch::msort(reshaped);
            }
        }

        // Test with different data types if possible
        if (input_tensor.dtype() != torch::kFloat32) {
            try {
                auto float_tensor = input_tensor.to(torch::kFloat32);
                auto result_float = torch::msort(float_tensor);
            } catch (...) {
                // Ignore conversion errors
            }
        }

        // Test with cloned tensor to ensure no memory issues
        auto cloned_tensor = input_tensor.clone();
        auto result_cloned = torch::msort(cloned_tensor);

        // Test with contiguous and non-contiguous tensors
        if (input_tensor.dim() >= 2) {
            auto transposed = input_tensor.transpose(0, 1);
            if (!transposed.is_contiguous()) {
                auto result_non_contiguous = torch::msort(transposed);
            }
        }

        // Test with special values if floating point
        if (input_tensor.dtype().is_floating_point()) {
            // Create tensor with special values
            auto special_tensor = torch::tensor({
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f, -0.0f, 1.0f, -1.0f
            });
            
            if (special_tensor.numel() > 0) {
                auto result_special = torch::msort(special_tensor);
            }
        }

        // Verify sorting property: result should be sorted along dim 0
        if (result1.dim() > 0 && result1.size(0) > 1) {
            // For each column (if 2D+), verify it's sorted
            if (result1.dim() == 1) {
                // Check if 1D tensor is sorted
                for (int64_t i = 0; i < result1.size(0) - 1; ++i) {
                    auto curr = result1[i];
                    auto next = result1[i + 1];
                    // Skip NaN comparisons as they have special sorting behavior
                    if (!torch::isnan(curr).item<bool>() && !torch::isnan(next).item<bool>()) {
                        if (torch::gt(curr, next).item<bool>()) {
                            std::cerr << "msort result not properly sorted at index " << i << std::endl;
                        }
                    }
                }
            }
        }

        // Test memory efficiency - ensure no unnecessary copies
        auto original_data_ptr = input_tensor.data_ptr();
        auto result_data_ptr = result1.data_ptr();
        // They should be different (msort creates new tensor)
        if (original_data_ptr == result_data_ptr && input_tensor.numel() > 0) {
            std::cerr << "msort may have returned input tensor instead of creating new one" << std::endl;
        }

        // Test with zero-sized tensors
        auto zero_tensor = torch::empty({0});
        auto result_zero = torch::msort(zero_tensor);

        // Test with single element tensor
        if (input_tensor.numel() > 0) {
            auto single_elem = input_tensor.flatten()[0].unsqueeze(0);
            auto result_single = torch::msort(single_elem);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}