#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor creation and dimension indices
        if (Size < 16) {
            return 0;
        }

        // Generate tensor dimensions (1-6 dimensions)
        int num_dims = (Data[offset] % 6) + 1;
        offset++;

        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims; i++) {
            if (offset >= Size) return 0;
            int64_t dim_size = (Data[offset] % 10) + 1; // 1-10 size per dimension
            dims.push_back(dim_size);
            offset++;
        }

        // Create tensor with random data
        torch::Tensor tensor = torch::randn(dims);

        // Test different data types
        if (offset < Size) {
            int dtype_choice = Data[offset] % 4;
            offset++;
            
            switch (dtype_choice) {
                case 0: tensor = tensor.to(torch::kFloat32); break;
                case 1: tensor = tensor.to(torch::kFloat64); break;
                case 2: tensor = tensor.to(torch::kInt32); break;
                case 3: tensor = tensor.to(torch::kInt64); break;
            }
        }

        // Generate dimension indices for swapping
        if (offset + 1 >= Size) return 0;
        
        int64_t dim0_raw = Data[offset];
        int64_t dim1_raw = Data[offset + 1];
        offset += 2;

        // Convert to valid dimension indices (both positive and negative)
        int64_t dim0 = (dim0_raw % (2 * num_dims)) - num_dims; // Range: [-num_dims, num_dims-1]
        int64_t dim1 = (dim1_raw % (2 * num_dims)) - num_dims;

        // Test torch::swapdims with various combinations
        torch::Tensor result1 = torch::swapdims(tensor, dim0, dim1);
        
        // Test with the same dimension (should return original tensor)
        torch::Tensor result2 = torch::swapdims(tensor, dim0, dim0);
        
        // Test swapping back (should restore original shape)
        torch::Tensor result3 = torch::swapdims(result1, dim0, dim1);
        
        // Test with boundary dimensions
        if (num_dims > 1) {
            torch::Tensor result4 = torch::swapdims(tensor, 0, num_dims - 1);
            torch::Tensor result5 = torch::swapdims(tensor, -1, -num_dims);
        }

        // Test edge cases with different tensor shapes
        if (offset < Size) {
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0});
            if (empty_tensor.dim() > 0) {
                torch::Tensor empty_result = torch::swapdims(empty_tensor, 0, 0);
            }

            // Test with scalar tensor (0-dimensional)
            torch::Tensor scalar = torch::tensor(42.0);
            // Note: swapdims on scalar should handle gracefully or throw

            // Test with 1D tensor
            torch::Tensor tensor_1d = torch::randn({10});
            torch::Tensor result_1d = torch::swapdims(tensor_1d, 0, 0);

            // Test with very large dimension numbers (should be handled by PyTorch)
            if (num_dims > 2) {
                int large_dim = 1000;
                // This should either work with modular arithmetic or throw an exception
                try {
                    torch::Tensor large_dim_result = torch::swapdims(tensor, 0, large_dim);
                } catch (...) {
                    // Expected for invalid dimensions
                }
            }
        }

        // Verify basic properties
        if (result1.numel() != tensor.numel()) {
            std::cerr << "Element count mismatch after swapdims" << std::endl;
        }

        if (result1.dtype() != tensor.dtype()) {
            std::cerr << "Data type changed after swapdims" << std::endl;
        }

        // Test method version vs function version
        torch::Tensor method_result = tensor.swapdims(dim0, dim1);
        
        // Test with different tensor layouts if possible
        if (tensor.dim() >= 2 && offset < Size) {
            torch::Tensor contiguous_tensor = tensor.contiguous();
            torch::Tensor non_contiguous = tensor.transpose(0, 1);
            
            torch::Tensor cont_result = torch::swapdims(contiguous_tensor, dim0, dim1);
            torch::Tensor non_cont_result = torch::swapdims(non_contiguous, dim0, dim1);
        }

        // Test with requires_grad tensor
        if (tensor.dtype().is_floating_point() && offset < Size) {
            torch::Tensor grad_tensor = tensor.clone().requires_grad_(true);
            torch::Tensor grad_result = torch::swapdims(grad_tensor, dim0, dim1);
            
            if (grad_result.requires_grad() != grad_tensor.requires_grad()) {
                std::cerr << "Gradient requirement not preserved" << std::endl;
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