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

        // Extract dimensions for first matrix
        int64_t m = extract_int64_t(Data, Size, offset) % 100 + 1; // rows of first matrix
        int64_t k = extract_int64_t(Data, Size, offset) % 100 + 1; // cols of first matrix / rows of second matrix
        int64_t n = extract_int64_t(Data, Size, offset) % 100 + 1; // cols of second matrix

        // Extract dtype information
        auto dtype1 = extract_dtype(Data, Size, offset);
        auto dtype2 = extract_dtype(Data, Size, offset);

        // Extract device information
        auto device1 = extract_device(Data, Size, offset);
        auto device2 = extract_device(Data, Size, offset);

        // Extract layout information
        auto layout1 = extract_layout(Data, Size, offset);
        auto layout2 = extract_layout(Data, Size, offset);

        // Create tensor options
        auto options1 = torch::TensorOptions().dtype(dtype1).device(device1).layout(layout1);
        auto options2 = torch::TensorOptions().dtype(dtype2).device(device2).layout(layout2);

        // Create first matrix (m x k)
        torch::Tensor mat1;
        if (layout1 == torch::kSparse) {
            // For sparse tensors, create a sparse tensor
            auto indices = torch::randint(0, std::max(m, k), {2, std::min(m * k / 4, (int64_t)10)}, torch::kLong);
            auto values = torch::randn({indices.size(1)}, options1.layout(torch::kStrided));
            mat1 = torch::sparse_coo_tensor(indices, values, {m, k}, options1);
        } else {
            mat1 = extract_tensor(Data, Size, offset, {m, k}, options1);
        }

        // Create second matrix (k x n)
        torch::Tensor mat2;
        if (layout2 == torch::kSparse) {
            // For sparse tensors, create a sparse tensor
            auto indices = torch::randint(0, std::max(k, n), {2, std::min(k * n / 4, (int64_t)10)}, torch::kLong);
            auto values = torch::randn({indices.size(1)}, options2.layout(torch::kStrided));
            mat2 = torch::sparse_coo_tensor(indices, values, {k, n}, options2);
        } else {
            mat2 = extract_tensor(Data, Size, offset, {k, n}, options2);
        }

        // Test basic matrix multiplication
        auto result = torch::mm(mat1, mat2);

        // Verify result shape
        if (result.sizes() != torch::IntArrayRef({m, n})) {
            std::cerr << "Unexpected result shape" << std::endl;
        }

        // Test with transposed matrices
        if (mat1.layout() != torch::kSparse && mat2.layout() != torch::kSparse) {
            auto mat1_t = mat1.t();
            auto mat2_t = mat2.t();
            
            // Test mat1^T * mat2 (requires mat1^T to be k x m, mat2 to be k x n)
            if (mat1_t.size(1) == mat2.size(0)) {
                auto result_t1 = torch::mm(mat1_t, mat2);
            }
            
            // Test mat1 * mat2^T (requires mat1 to be m x k, mat2^T to be k x n)
            if (mat1.size(1) == mat2_t.size(0)) {
                auto result_t2 = torch::mm(mat1, mat2_t);
            }
        }

        // Test edge cases with different tensor properties
        if (offset < Size) {
            bool requires_grad1 = extract_bool(Data, Size, offset);
            bool requires_grad2 = extract_bool(Data, Size, offset);
            
            if (mat1.dtype().is_floating_point()) {
                mat1.requires_grad_(requires_grad1);
            }
            if (mat2.dtype().is_floating_point()) {
                mat2.requires_grad_(requires_grad2);
            }
            
            auto result_grad = torch::mm(mat1, mat2);
            
            // Test backward pass if gradients are required
            if (result_grad.requires_grad()) {
                auto grad_output = torch::ones_like(result_grad);
                result_grad.backward(grad_output);
            }
        }

        // Test with contiguous and non-contiguous tensors
        if (mat1.layout() != torch::kSparse && mat2.layout() != torch::kSparse) {
            auto mat1_nc = mat1.transpose(0, 1).transpose(0, 1); // Make non-contiguous
            auto mat2_nc = mat2.transpose(0, 1).transpose(0, 1); // Make non-contiguous
            
            if (mat1_nc.size(1) == mat2_nc.size(0)) {
                auto result_nc = torch::mm(mat1_nc, mat2_nc);
            }
        }

        // Test with zero-sized dimensions (edge case)
        if (offset < Size) {
            bool test_zero_dim = extract_bool(Data, Size, offset);
            if (test_zero_dim && m > 1 && n > 1) {
                auto zero_mat1 = torch::zeros({m, 0}, options1.layout(torch::kStrided));
                auto zero_mat2 = torch::zeros({0, n}, options2.layout(torch::kStrided));
                auto zero_result = torch::mm(zero_mat1, zero_mat2);
            }
        }

        // Test with very small matrices
        if (offset < Size) {
            bool test_small = extract_bool(Data, Size, offset);
            if (test_small) {
                auto small_mat1 = torch::randn({1, 1}, options1.layout(torch::kStrided));
                auto small_mat2 = torch::randn({1, 1}, options2.layout(torch::kStrided));
                auto small_result = torch::mm(small_mat1, small_mat2);
            }
        }

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