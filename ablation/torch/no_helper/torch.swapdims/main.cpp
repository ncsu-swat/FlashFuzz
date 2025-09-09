#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor shape, dimensions, and some tensor data
        if (Size < 16) {
            return 0;
        }

        // Generate tensor shape (1-6 dimensions to avoid memory issues)
        auto shape_info = generate_tensor_shape(Data, Size, offset, 1, 6);
        auto shape = shape_info.first;
        
        if (shape.empty()) {
            return 0;
        }

        // Create input tensor with random data
        torch::Tensor input = generate_random_tensor(Data, Size, offset, shape);
        
        if (offset >= Size - 8) {
            return 0;
        }

        // Extract dim0 and dim1 from remaining data
        int64_t dim0_raw = read_int64(Data, Size, offset);
        int64_t dim1_raw = read_int64(Data, Size, offset);
        
        int64_t ndim = static_cast<int64_t>(input.dim());
        
        // Handle edge cases for dimensions
        if (ndim == 0) {
            // 0-dimensional tensor - swapdims should handle this gracefully
            dim0_raw = 0;
            dim1_raw = 0;
        }
        
        // Normalize dimensions to valid range [-ndim, ndim-1]
        int64_t dim0 = ((dim0_raw % (2 * ndim)) + (2 * ndim)) % (2 * ndim) - ndim;
        int64_t dim1 = ((dim1_raw % (2 * ndim)) + (2 * ndim)) % (2 * ndim) - ndim;
        
        // Clamp to valid range
        dim0 = std::max(-ndim, std::min(ndim - 1, dim0));
        dim1 = std::max(-ndim, std::min(ndim - 1, dim1));

        // Test basic swapdims operation
        torch::Tensor result = torch::swapdims(input, dim0, dim1);
        
        // Verify result properties
        if (result.dim() != input.dim()) {
            std::cerr << "Dimension count mismatch after swapdims" << std::endl;
        }
        
        // Test edge cases
        
        // 1. Same dimension (should be identity)
        torch::Tensor same_dim_result = torch::swapdims(input, dim0, dim0);
        
        // 2. Negative dimension indexing
        if (ndim > 0) {
            int64_t neg_dim0 = dim0 < 0 ? dim0 : dim0 - ndim;
            int64_t neg_dim1 = dim1 < 0 ? dim1 : dim1 - ndim;
            torch::Tensor neg_result = torch::swapdims(input, neg_dim0, neg_dim1);
        }
        
        // 3. Test with different tensor types if we have enough data
        if (offset < Size - 4) {
            uint32_t type_selector = read_uint32(Data, Size, offset) % 4;
            torch::Tensor typed_input;
            
            switch (type_selector) {
                case 0:
                    typed_input = input.to(torch::kFloat32);
                    break;
                case 1:
                    typed_input = input.to(torch::kFloat64);
                    break;
                case 2:
                    typed_input = input.to(torch::kInt32);
                    break;
                case 3:
                    typed_input = input.to(torch::kInt64);
                    break;
            }
            
            torch::Tensor typed_result = torch::swapdims(typed_input, dim0, dim1);
        }
        
        // 4. Test with contiguous and non-contiguous tensors
        if (input.numel() > 1) {
            // Create a non-contiguous tensor by transposing first
            if (ndim >= 2) {
                torch::Tensor non_contiguous = input.transpose(0, 1);
                torch::Tensor non_cont_result = torch::swapdims(non_contiguous, dim0, dim1);
            }
        }
        
        // 5. Test boundary dimensions
        if (ndim > 1) {
            torch::Tensor boundary_result1 = torch::swapdims(input, 0, ndim - 1);
            torch::Tensor boundary_result2 = torch::swapdims(input, -1, -ndim);
        }
        
        // 6. Test with empty tensor
        if (input.numel() == 0 && ndim > 0) {
            torch::Tensor empty_result = torch::swapdims(input, dim0, dim1);
        }
        
        // 7. Verify swapdims is equivalent to transpose
        torch::Tensor transpose_result = torch::transpose(input, dim0, dim1);
        if (!torch::allclose(result, transpose_result, 1e-6, 1e-6, /*equal_nan=*/true)) {
            std::cerr << "swapdims result differs from transpose" << std::endl;
        }
        
        // 8. Test double swap (should return to original for different dims)
        if (dim0 != dim1) {
            torch::Tensor double_swap = torch::swapdims(result, dim0, dim1);
            if (!torch::allclose(input, double_swap, 1e-6, 1e-6, /*equal_nan=*/true)) {
                std::cerr << "Double swapdims did not return to original" << std::endl;
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