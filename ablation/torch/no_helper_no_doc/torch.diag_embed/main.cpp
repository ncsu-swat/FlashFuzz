#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation and parameters
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions and properties
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        if (tensor_info.dims.empty()) {
            return 0;
        }

        // Create input tensor - diag_embed works on tensors of any dimension
        torch::Tensor input = create_tensor(tensor_info);
        if (!input.defined()) {
            return 0;
        }

        // Extract parameters for diag_embed
        int8_t offset_param = extract_value<int8_t>(Data, Size, offset);
        int8_t dim1_param = extract_value<int8_t>(Data, Size, offset);
        int8_t dim2_param = extract_value<int8_t>(Data, Size, offset);

        // Clamp offset parameter to reasonable range
        int offset_val = static_cast<int>(offset_param) % 10;

        // Get input tensor dimensions for dim parameter validation
        int64_t input_ndim = input.dim();
        
        // Clamp dim1 and dim2 to valid ranges
        // For diag_embed, dim1 and dim2 should be in range [-output_ndim, output_ndim-1]
        // where output_ndim = input_ndim + 1
        int64_t output_ndim = input_ndim + 1;
        int dim1_val = static_cast<int>(dim1_param) % static_cast<int>(output_ndim * 2) - static_cast<int>(output_ndim);
        int dim2_val = static_cast<int>(dim2_param) % static_cast<int>(output_ndim * 2) - static_cast<int>(output_ndim);

        // Ensure dim1 != dim2
        if (dim1_val == dim2_val) {
            dim2_val = (dim2_val + 1) % static_cast<int>(output_ndim);
            if (dim2_val == dim1_val) {
                dim2_val = (dim2_val + 1) % static_cast<int>(output_ndim);
            }
        }

        // Test basic diag_embed
        torch::Tensor result1 = torch::diag_embed(input);

        // Test diag_embed with offset
        torch::Tensor result2 = torch::diag_embed(input, offset_val);

        // Test diag_embed with offset and dim1
        torch::Tensor result3 = torch::diag_embed(input, offset_val, dim1_val);

        // Test diag_embed with all parameters
        torch::Tensor result4 = torch::diag_embed(input, offset_val, dim1_val, dim2_val);

        // Test edge cases with different tensor types
        if (input.numel() > 0) {
            // Test with different dtypes if possible
            if (input.dtype() != torch::kBool) {
                auto bool_input = input.to(torch::kBool);
                torch::Tensor bool_result = torch::diag_embed(bool_input);
            }

            if (input.dtype() != torch::kFloat32) {
                auto float_input = input.to(torch::kFloat32);
                torch::Tensor float_result = torch::diag_embed(float_input, offset_val);
            }

            if (input.dtype() != torch::kInt64) {
                auto int_input = input.to(torch::kInt64);
                torch::Tensor int_result = torch::diag_embed(int_input, offset_val, dim1_val, dim2_val);
            }
        }

        // Test with zero-sized tensors
        if (input.numel() == 0) {
            torch::Tensor empty_result = torch::diag_embed(input);
        }

        // Test with 1D tensors (common case)
        if (input.dim() > 1) {
            auto flattened = input.flatten();
            torch::Tensor flat_result = torch::diag_embed(flattened, offset_val);
        }

        // Test with very small tensors
        if (input.numel() > 1) {
            auto small_input = input.narrow(input.dim() - 1, 0, 1);
            torch::Tensor small_result = torch::diag_embed(small_input);
        }

        // Verify results are valid tensors
        if (!result1.defined() || !result2.defined() || !result3.defined() || !result4.defined()) {
            return -1;
        }

        // Basic sanity checks on output dimensions
        if (result1.dim() != input.dim() + 1) {
            return -1;
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && input.numel() < 1000) {
            try {
                auto cuda_input = input.to(torch::kCUDA);
                torch::Tensor cuda_result = torch::diag_embed(cuda_input, offset_val, dim1_val, dim2_val);
                if (!cuda_result.defined()) {
                    return -1;
                }
            } catch (...) {
                // CUDA operations might fail, ignore
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