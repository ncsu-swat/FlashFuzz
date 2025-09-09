#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for two matrices
        if (Size < 16) return 0;

        // Extract dimensions for first matrix (input)
        auto input_rows = extract_int(Data, Size, offset, 1, 100);
        auto input_cols = extract_int(Data, Size, offset, 1, 100);
        
        // Extract dimensions for second matrix (mat2)
        // mat2 rows must equal input_cols for valid matrix multiplication
        auto mat2_rows = input_cols; // This ensures compatibility
        auto mat2_cols = extract_int(Data, Size, offset, 1, 100);

        // Extract data types
        auto dtype1 = extract_dtype(Data, Size, offset);
        auto dtype2 = extract_dtype(Data, Size, offset);

        // Extract device type
        auto device = extract_device(Data, Size, offset);

        // Extract layout types
        auto layout1 = extract_layout(Data, Size, offset);
        auto layout2 = extract_layout(Data, Size, offset);

        // Extract whether to use output tensor
        bool use_out = extract_bool(Data, Size, offset);

        // Create tensor options
        auto options1 = torch::TensorOptions().dtype(dtype1).device(device).layout(layout1);
        auto options2 = torch::TensorOptions().dtype(dtype2).device(device).layout(layout2);

        // Create input tensors
        torch::Tensor input, mat2;

        if (layout1 == torch::kSparse) {
            // Create sparse tensor for input
            auto indices = torch::randint(0, std::max(input_rows, input_cols), {2, std::min(10, input_rows * input_cols / 4)}, torch::kLong);
            auto values = torch::randn({indices.size(1)}, options1.layout(torch::kStrided));
            input = torch::sparse_coo_tensor(indices, values, {input_rows, input_cols}, options1);
        } else {
            input = torch::randn({input_rows, input_cols}, options1);
        }

        if (layout2 == torch::kSparse) {
            // Create sparse tensor for mat2
            auto indices = torch::randint(0, std::max(mat2_rows, mat2_cols), {2, std::min(10, mat2_rows * mat2_cols / 4)}, torch::kLong);
            auto values = torch::randn({indices.size(1)}, options2.layout(torch::kStrided));
            mat2 = torch::sparse_coo_tensor(indices, values, {mat2_rows, mat2_cols}, options2);
        } else {
            mat2 = torch::randn({mat2_rows, mat2_cols}, options2);
        }

        // Test basic matrix multiplication
        auto result = torch::mm(input, mat2);

        // Verify result shape
        if (result.size(0) != input_rows || result.size(1) != mat2_cols) {
            throw std::runtime_error("Result shape mismatch");
        }

        // Test with output tensor if requested
        if (use_out) {
            auto out_tensor = torch::empty({input_rows, mat2_cols}, 
                                         torch::TensorOptions().dtype(result.dtype()).device(device));
            torch::mm_out(out_tensor, input, mat2);
            
            // Verify output tensor was modified
            if (!torch::allclose(result, out_tensor, 1e-5, 1e-8, /*equal_nan=*/false)) {
                // Allow for small numerical differences
            }
        }

        // Test edge cases with different tensor properties
        if (offset < Size) {
            // Test with transposed inputs
            bool transpose_input = extract_bool(Data, Size, offset);
            bool transpose_mat2 = extract_bool(Data, Size, offset);
            
            if (transpose_input && input.dim() == 2) {
                auto transposed_input = input.t();
                if (transposed_input.size(1) == mat2.size(0)) {
                    auto result_t = torch::mm(transposed_input, mat2);
                }
            }
            
            if (transpose_mat2 && mat2.dim() == 2) {
                auto transposed_mat2 = mat2.t();
                if (input.size(1) == transposed_mat2.size(0)) {
                    auto result_t = torch::mm(input, transposed_mat2);
                }
            }
        }

        // Test with different strides if possible
        if (offset < Size && layout1 == torch::kStrided && layout2 == torch::kStrided) {
            bool test_strided = extract_bool(Data, Size, offset);
            if (test_strided) {
                // Create strided versions
                auto strided_input = input.as_strided({input_rows, input_cols}, {input_cols, 1});
                auto strided_mat2 = mat2.as_strided({mat2_rows, mat2_cols}, {mat2_cols, 1});
                auto strided_result = torch::mm(strided_input, strided_mat2);
            }
        }

        // Test autograd if tensors require gradients
        if (offset < Size) {
            bool test_grad = extract_bool(Data, Size, offset);
            if (test_grad && input.dtype().is_floating_point() && mat2.dtype().is_floating_point()) {
                auto grad_input = input.clone().requires_grad_(true);
                auto grad_mat2 = mat2.clone().requires_grad_(true);
                
                auto grad_result = torch::mm(grad_input, grad_mat2);
                auto loss = grad_result.sum();
                loss.backward();
                
                // Check that gradients exist
                if (grad_input.grad().defined() && grad_mat2.grad().defined()) {
                    // Gradients computed successfully
                }
            }
        }

        // Test with special values if floating point
        if (offset < Size && input.dtype().is_floating_point()) {
            bool test_special = extract_bool(Data, Size, offset);
            if (test_special) {
                // Test with tensors containing inf, -inf, nan
                auto special_input = input.clone();
                auto special_mat2 = mat2.clone();
                
                if (special_input.numel() > 0) {
                    special_input.flatten()[0] = std::numeric_limits<float>::infinity();
                }
                if (special_mat2.numel() > 0) {
                    special_mat2.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                }
                
                auto special_result = torch::mm(special_input, special_mat2);
            }
        }

        // Test memory layout preservation
        result.contiguous();
        
        // Force evaluation of lazy tensors
        result.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}