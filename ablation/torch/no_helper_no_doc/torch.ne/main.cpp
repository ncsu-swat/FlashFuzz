#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to work with
        if (Size < 16) {
            return 0;
        }

        // Parse tensor shapes and dtypes
        auto shape1 = parse_shape(Data, Size, offset);
        auto shape2 = parse_shape(Data, Size, offset);
        auto dtype1 = parse_dtype(Data, Size, offset);
        auto dtype2 = parse_dtype(Data, Size, offset);

        // Create tensors with different data types to test type promotion
        auto tensor1 = create_tensor(shape1, dtype1, Data, Size, offset);
        auto tensor2 = create_tensor(shape2, dtype2, Data, Size, offset);

        // Test torch.ne with two tensors
        auto result1 = torch::ne(tensor1, tensor2);

        // Test broadcasting scenarios
        if (tensor1.numel() > 0 && tensor2.numel() > 0) {
            try {
                auto broadcasted_result = torch::ne(tensor1, tensor2);
            } catch (const std::exception&) {
                // Broadcasting might fail, that's okay
            }
        }

        // Test torch.ne with tensor and scalar
        if (offset < Size - 8) {
            double scalar_val = parse_scalar<double>(Data, Size, offset);
            auto result2 = torch::ne(tensor1, scalar_val);
            
            // Test with different scalar types
            int64_t int_scalar = static_cast<int64_t>(scalar_val);
            auto result3 = torch::ne(tensor1, int_scalar);
            
            float float_scalar = static_cast<float>(scalar_val);
            auto result4 = torch::ne(tensor1, float_scalar);
        }

        // Test with special values
        if (tensor1.dtype().isFloatingPoint()) {
            auto nan_result = torch::ne(tensor1, std::numeric_limits<double>::quiet_NaN());
            auto inf_result = torch::ne(tensor1, std::numeric_limits<double>::infinity());
            auto neg_inf_result = torch::ne(tensor1, -std::numeric_limits<double>::infinity());
        }

        // Test with zero tensors
        auto zero_tensor = torch::zeros_like(tensor1);
        auto zero_result = torch::ne(tensor1, zero_tensor);

        // Test with same tensor (should return all false)
        auto self_result = torch::ne(tensor1, tensor1);

        // Test edge cases with empty tensors
        if (tensor1.numel() == 0 || tensor2.numel() == 0) {
            try {
                auto empty_result = torch::ne(tensor1, tensor2);
            } catch (const std::exception&) {
                // Empty tensor operations might fail in some cases
            }
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && tensor1.numel() > 0) {
            try {
                auto cuda_tensor1 = tensor1.to(torch::kCUDA);
                auto cuda_result = torch::ne(cuda_tensor1, cuda_tensor1);
                
                // Test CPU vs CUDA comparison (should move to same device)
                if (tensor2.numel() > 0) {
                    auto mixed_result = torch::ne(cuda_tensor1, tensor2.to(torch::kCUDA));
                }
            } catch (const std::exception&) {
                // CUDA operations might fail, that's okay
            }
        }

        // Test with complex numbers if supported
        if (offset < Size - 16) {
            try {
                auto complex_tensor = torch::complex(tensor1.to(torch::kFloat), tensor1.to(torch::kFloat));
                auto complex_result = torch::ne(complex_tensor, complex_tensor);
            } catch (const std::exception&) {
                // Complex operations might fail with certain dtypes
            }
        }

        // Test with boolean tensors
        try {
            auto bool_tensor1 = tensor1.to(torch::kBool);
            auto bool_tensor2 = tensor2.to(torch::kBool);
            auto bool_result = torch::ne(bool_tensor1, bool_tensor2);
        } catch (const std::exception&) {
            // Boolean conversion might fail
        }

        // Test output tensor variant if we have enough data
        if (offset < Size - 4) {
            try {
                auto out_tensor = torch::empty_like(tensor1, torch::kBool);
                torch::ne_out(out_tensor, tensor1, tensor2);
            } catch (const std::exception&) {
                // out variant might fail due to shape/type mismatches
            }
        }

        // Test with very large and very small values
        if (tensor1.dtype().isFloatingPoint() && tensor1.numel() > 0) {
            auto large_val_result = torch::ne(tensor1, 1e20);
            auto small_val_result = torch::ne(tensor1, 1e-20);
        }

        // Test chained comparisons
        if (tensor1.numel() > 0 && tensor2.numel() > 0) {
            try {
                auto chained = torch::ne(torch::ne(tensor1, tensor2), torch::ne(tensor2, tensor1));
            } catch (const std::exception&) {
                // Chained operations might fail
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}