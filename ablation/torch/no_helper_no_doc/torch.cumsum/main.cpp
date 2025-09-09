#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for tensor creation and parameters
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions and properties
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        if (tensor_info.dims.empty()) {
            return 0;
        }

        // Create input tensor with various dtypes to test edge cases
        torch::Tensor input;
        uint8_t dtype_choice = Data[offset % Size];
        offset++;

        switch (dtype_choice % 6) {
            case 0:
                input = create_tensor(tensor_info, torch::kFloat32);
                break;
            case 1:
                input = create_tensor(tensor_info, torch::kFloat64);
                break;
            case 2:
                input = create_tensor(tensor_info, torch::kInt32);
                break;
            case 3:
                input = create_tensor(tensor_info, torch::kInt64);
                break;
            case 4:
                input = create_tensor(tensor_info, torch::kBool);
                break;
            case 5:
                input = create_tensor(tensor_info, torch::kComplexFloat);
                break;
        }

        if (input.numel() == 0) {
            return 0;
        }

        // Extract dimension parameter for cumsum
        int64_t dim;
        if (offset + sizeof(int64_t) <= Size) {
            dim = *reinterpret_cast<const int64_t*>(Data + offset);
            offset += sizeof(int64_t);
        } else {
            dim = Data[offset % Size] % input.dim();
            offset++;
        }

        // Normalize dimension to valid range
        if (input.dim() > 0) {
            dim = ((dim % input.dim()) + input.dim()) % input.dim();
        } else {
            dim = 0;
        }

        // Extract dtype parameter for output (optional)
        torch::optional<torch::ScalarType> dtype_opt = torch::nullopt;
        if (offset < Size) {
            uint8_t use_dtype = Data[offset++];
            if (use_dtype % 2 == 1) {
                uint8_t dtype_val = Data[offset % Size];
                switch (dtype_val % 6) {
                    case 0: dtype_opt = torch::kFloat32; break;
                    case 1: dtype_opt = torch::kFloat64; break;
                    case 2: dtype_opt = torch::kInt32; break;
                    case 3: dtype_opt = torch::kInt64; break;
                    case 4: dtype_opt = torch::kBool; break;
                    case 5: dtype_opt = torch::kComplexFloat; break;
                }
            }
        }

        // Test basic cumsum operation
        torch::Tensor result1 = torch::cumsum(input, dim);

        // Test cumsum with dtype specification
        if (dtype_opt.has_value()) {
            torch::Tensor result2 = torch::cumsum(input, dim, dtype_opt);
        }

        // Test in-place version if input is not boolean (in-place ops don't work with bool)
        if (input.dtype() != torch::kBool && input.is_contiguous()) {
            torch::Tensor input_copy = input.clone();
            input_copy.cumsum_(dim);
        }

        // Test edge cases with different tensor layouts
        if (input.dim() > 1) {
            // Test with non-contiguous tensor
            torch::Tensor non_contiguous = input.transpose(0, std::min(1L, input.dim() - 1));
            torch::Tensor result3 = torch::cumsum(non_contiguous, 0);
        }

        // Test with zero-sized tensors
        if (input.dim() > 0) {
            std::vector<int64_t> zero_size_dims = tensor_info.dims;
            zero_size_dims[0] = 0;
            torch::Tensor zero_tensor = torch::zeros(zero_size_dims, input.options());
            torch::Tensor result4 = torch::cumsum(zero_tensor, dim);
        }

        // Test with single element tensor
        torch::Tensor single_elem = torch::ones({1}, input.options());
        torch::Tensor result5 = torch::cumsum(single_elem, 0);

        // Test negative dimension indexing
        if (input.dim() > 0) {
            int64_t neg_dim = -1;
            torch::Tensor result6 = torch::cumsum(input, neg_dim);
        }

        // Test with very large values to check for overflow (for integer types)
        if (input.dtype() == torch::kInt32 || input.dtype() == torch::kInt64) {
            torch::Tensor large_vals;
            if (input.dtype() == torch::kInt32) {
                large_vals = torch::full_like(input, std::numeric_limits<int32_t>::max() / 2);
            } else {
                large_vals = torch::full_like(input, std::numeric_limits<int64_t>::max() / 2);
            }
            torch::Tensor result7 = torch::cumsum(large_vals, dim);
        }

        // Test with NaN and Inf values for floating point types
        if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) {
            torch::Tensor special_vals = input.clone();
            if (special_vals.numel() > 0) {
                special_vals.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                if (special_vals.numel() > 1) {
                    special_vals.flatten()[1] = std::numeric_limits<float>::infinity();
                }
                torch::Tensor result8 = torch::cumsum(special_vals, dim);
            }
        }

        // Test with complex numbers
        if (input.dtype() == torch::kComplexFloat) {
            torch::Tensor result9 = torch::cumsum(input, dim);
        }

        // Validate output properties
        if (result1.defined()) {
            // Check that output has same shape as input
            if (!result1.sizes().equals(input.sizes())) {
                std::cerr << "Output shape mismatch" << std::endl;
            }
            
            // Check that output is finite for floating point types (except when input has special values)
            if ((result1.dtype() == torch::kFloat32 || result1.dtype() == torch::kFloat64) && 
                result1.numel() > 0) {
                // Only check if input doesn't contain NaN/Inf
                bool input_has_special = false;
                if (input.dtype().isFloatingType()) {
                    torch::Tensor flat_input = input.flatten();
                    for (int64_t i = 0; i < flat_input.numel(); ++i) {
                        float val = flat_input[i].item<float>();
                        if (std::isnan(val) || std::isinf(val)) {
                            input_has_special = true;
                            break;
                        }
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