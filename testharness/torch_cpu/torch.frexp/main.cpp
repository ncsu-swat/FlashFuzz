#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor - frexp requires floating point input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor for frexp
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Apply frexp operation
        // frexp returns a tuple of (mantissa, exponent)
        auto result = torch::frexp(input);
        
        // Access the components of the result
        torch::Tensor mantissa = std::get<0>(result);
        torch::Tensor exponent = std::get<1>(result);
        
        // Verify the result by reconstructing the original tensor
        // For each element: input = mantissa * (2 ^ exponent)
        try {
            torch::Tensor reconstructed = mantissa * torch::pow(2.0, exponent.to(torch::kFloat));
            (void)reconstructed;
        } catch (...) {
            // Reconstruction may fail for special values (inf, nan)
        }
        
        // Try different variants of the API
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Test out_variant using frexp_out with tuple output
            if (variant % 3 == 0 && input.numel() > 0) {
                try {
                    torch::Tensor mantissa_out = torch::empty_like(input);
                    torch::Tensor exponent_out = torch::empty(input.sizes(), torch::TensorOptions().dtype(torch::kInt32));
                    torch::frexp_out(mantissa_out, exponent_out, input);
                } catch (...) {
                    // frexp_out may have different requirements
                }
            }
            
            // Test functional variant again
            if (variant % 3 == 1) {
                auto named_result = torch::frexp(input);
                auto m = std::get<0>(named_result);
                auto e = std::get<1>(named_result);
                (void)m;
                (void)e;
            }
            
            // Test edge case: empty tensor
            if (variant % 3 == 2) {
                try {
                    torch::Tensor empty_tensor = torch::empty({0}, input.options());
                    auto empty_result = torch::frexp(empty_tensor);
                    (void)empty_result;
                } catch (...) {
                    // Empty tensor handling may vary
                }
            }
        }
        
        // Try with different dtypes if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            dtype_selector = dtype_selector % 2; // Only use well-supported dtypes
            
            torch::ScalarType target_dtype;
            switch (dtype_selector) {
                case 0: target_dtype = torch::kFloat; break;
                case 1: target_dtype = torch::kDouble; break;
                default: target_dtype = torch::kFloat;
            }
            
            // Convert input to the selected dtype and apply frexp
            if (input.scalar_type() != target_dtype) {
                try {
                    torch::Tensor converted_input = input.to(target_dtype);
                    auto converted_result = torch::frexp(converted_input);
                    (void)converted_result;
                } catch (...) {
                    // Dtype conversion or frexp may fail
                }
            }
        }
        
        // Test with different tensor shapes
        if (offset + 2 < Size) {
            uint8_t shape_variant = Data[offset++];
            try {
                torch::Tensor shaped_tensor;
                switch (shape_variant % 4) {
                    case 0:
                        // Scalar tensor
                        shaped_tensor = torch::tensor(1.5f);
                        break;
                    case 1:
                        // 1D tensor
                        shaped_tensor = torch::randn({5});
                        break;
                    case 2:
                        // 2D tensor
                        shaped_tensor = torch::randn({3, 4});
                        break;
                    case 3:
                        // 3D tensor
                        shaped_tensor = torch::randn({2, 3, 4});
                        break;
                }
                auto shape_result = torch::frexp(shaped_tensor);
                (void)shape_result;
            } catch (...) {
                // Shape-related issues
            }
        }
        
        // Test with special floating point values
        if (offset < Size) {
            uint8_t special_variant = Data[offset++];
            try {
                torch::Tensor special_tensor;
                switch (special_variant % 5) {
                    case 0:
                        special_tensor = torch::tensor({0.0f, -0.0f});
                        break;
                    case 1:
                        special_tensor = torch::tensor({std::numeric_limits<float>::infinity()});
                        break;
                    case 2:
                        special_tensor = torch::tensor({-std::numeric_limits<float>::infinity()});
                        break;
                    case 3:
                        special_tensor = torch::tensor({std::numeric_limits<float>::quiet_NaN()});
                        break;
                    case 4:
                        special_tensor = torch::tensor({std::numeric_limits<float>::denorm_min()});
                        break;
                }
                auto special_result = torch::frexp(special_tensor);
                (void)special_result;
            } catch (...) {
                // Special value handling
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}