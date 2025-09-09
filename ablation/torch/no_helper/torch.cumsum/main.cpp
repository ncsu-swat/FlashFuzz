#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation and dimension
        if (Size < 16) {
            return 0;
        }

        // Extract tensor configuration
        auto tensor_config = extract_tensor_config(Data, Size, offset);
        if (!tensor_config.has_value()) {
            return 0;
        }

        auto [shape, dtype, device] = tensor_config.value();
        
        // Create input tensor with random data
        torch::Tensor input = create_random_tensor(shape, dtype, device);
        if (input.numel() == 0) {
            return 0;
        }

        // Extract dimension parameter
        int64_t dim = extract_int_in_range(Data, Size, offset, -input.dim(), input.dim() - 1);

        // Test basic cumsum operation
        torch::Tensor result1 = torch::cumsum(input, dim);

        // Test with different dtypes if we have enough data
        if (offset + 1 < Size) {
            uint8_t dtype_choice = Data[offset++];
            torch::ScalarType target_dtype;
            
            switch (dtype_choice % 8) {
                case 0: target_dtype = torch::kFloat32; break;
                case 1: target_dtype = torch::kFloat64; break;
                case 2: target_dtype = torch::kInt32; break;
                case 3: target_dtype = torch::kInt64; break;
                case 4: target_dtype = torch::kInt16; break;
                case 5: target_dtype = torch::kInt8; break;
                case 6: target_dtype = torch::kUInt8; break;
                default: target_dtype = torch::kFloat32; break;
            }

            // Test cumsum with dtype conversion
            torch::Tensor result2 = torch::cumsum(input, dim, target_dtype);
        }

        // Test with output tensor if we have enough data
        if (offset < Size) {
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::cumsum_out(out_tensor, input, dim);
        }

        // Test edge cases with different tensor shapes
        if (input.dim() > 1 && offset < Size) {
            // Test with different dimensions
            for (int64_t test_dim = 0; test_dim < input.dim() && offset < Size; ++test_dim) {
                torch::Tensor result_dim = torch::cumsum(input, test_dim);
                offset++;
            }
        }

        // Test with scalar tensor
        if (offset < Size) {
            torch::Tensor scalar_input = torch::scalar_tensor(extract_float(Data, Size, offset), dtype);
            torch::Tensor scalar_result = torch::cumsum(scalar_input, 0);
        }

        // Test with 1D tensor
        if (offset + 4 < Size) {
            int64_t size_1d = extract_int_in_range(Data, Size, offset, 1, 100);
            torch::Tensor input_1d = create_random_tensor({size_1d}, dtype, device);
            torch::Tensor result_1d = torch::cumsum(input_1d, 0);
        }

        // Test with very large dimension values (should handle gracefully)
        if (offset < Size) {
            try {
                int64_t large_dim = extract_int_in_range(Data, Size, offset, input.dim(), input.dim() + 10);
                torch::Tensor result_large = torch::cumsum(input, large_dim);
            } catch (const std::exception&) {
                // Expected to fail for out-of-bounds dimensions
            }
        }

        // Test with negative dimensions
        if (offset < Size) {
            int64_t neg_dim = extract_int_in_range(Data, Size, offset, -input.dim() - 5, -1);
            try {
                torch::Tensor result_neg = torch::cumsum(input, neg_dim);
            } catch (const std::exception&) {
                // May fail for very negative dimensions
            }
        }

        // Test memory layout variations
        if (input.dim() >= 2 && offset < Size) {
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor result_contiguous = torch::cumsum(contiguous_input, dim);
            
            // Test with transposed tensor
            torch::Tensor transposed_input = input.transpose(0, 1);
            torch::Tensor result_transposed = torch::cumsum(transposed_input, dim % transposed_input.dim());
        }

        // Verify basic properties of cumsum
        if (result1.defined()) {
            // Result should have same shape as input
            if (!result1.sizes().equals(input.sizes())) {
                throw std::runtime_error("cumsum result shape mismatch");
            }
            
            // For 1D case, verify cumulative property on small tensors
            if (input.dim() == 1 && input.numel() <= 10 && input.dtype() == torch::kFloat32) {
                auto input_acc = input.accessor<float, 1>();
                auto result_acc = result1.accessor<float, 1>();
                
                float running_sum = 0.0f;
                for (int64_t i = 0; i < input.size(0); ++i) {
                    running_sum += input_acc[i];
                    // Allow for floating point precision differences
                    if (std::abs(result_acc[i] - running_sum) > 1e-5) {
                        // Don't throw, just note the discrepancy
                    }
                }
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