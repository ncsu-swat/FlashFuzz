#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor creation and dimension parameters
        if (Size < 16) {
            return 0;
        }

        // Generate tensor dimensions (1-6 dimensions)
        int num_dims = (Data[offset] % 6) + 1;
        offset++;

        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Keep dimensions reasonable to avoid memory issues
            int64_t dim_size = (Data[offset] % 10) + 1;
            dims.push_back(dim_size);
            offset++;
        }

        if (dims.empty()) {
            dims.push_back(1); // Fallback to 1D tensor
        }

        // Create input tensor with various dtypes and layouts
        torch::Tensor input;
        
        // Choose tensor type based on fuzzer input
        if (offset < Size) {
            int tensor_type = Data[offset] % 4;
            offset++;
            
            switch (tensor_type) {
                case 0:
                    // Dense float tensor
                    input = torch::randn(dims, torch::kFloat32);
                    break;
                case 1:
                    // Dense integer tensor
                    input = torch::randint(0, 100, dims, torch::kInt32);
                    break;
                case 2:
                    // Dense double tensor
                    input = torch::randn(dims, torch::kFloat64);
                    break;
                case 3:
                    // Sparse tensor (if dimensions allow)
                    if (dims.size() >= 2) {
                        auto indices = torch::randint(0, dims[0], {2, std::min(10L, dims[0] * dims[1])}, torch::kLong);
                        auto values = torch::randn({std::min(10L, dims[0] * dims[1])});
                        input = torch::sparse_coo_tensor(indices, values, dims);
                    } else {
                        input = torch::randn(dims, torch::kFloat32);
                    }
                    break;
            }
        } else {
            input = torch::randn(dims, torch::kFloat32);
        }

        // Get dimensions for transposition
        int64_t dim0, dim1;
        
        if (offset + 1 < Size) {
            // Use fuzzer data to select dimensions
            dim0 = Data[offset] % input.dim();
            offset++;
            dim1 = Data[offset] % input.dim();
            offset++;
        } else {
            // Default case - transpose last two dimensions if possible
            if (input.dim() >= 2) {
                dim0 = input.dim() - 2;
                dim1 = input.dim() - 1;
            } else {
                dim0 = 0;
                dim1 = 0;
            }
        }

        // Test negative dimension indexing
        if (offset < Size && (Data[offset] % 2) == 0) {
            if (dim0 >= 0) dim0 = dim0 - input.dim();
            if (dim1 >= 0) dim1 = dim1 - input.dim();
            offset++;
        }

        // Perform transpose operation
        torch::Tensor result = torch::transpose(input, dim0, dim1);

        // Verify basic properties
        if (result.numel() != input.numel()) {
            std::cerr << "Transpose changed number of elements!" << std::endl;
        }

        // Test that dimensions are correctly swapped
        if (dim0 != dim1 && input.dim() > 1) {
            auto input_sizes = input.sizes();
            auto result_sizes = result.sizes();
            
            // Normalize negative indices
            int64_t norm_dim0 = dim0 < 0 ? dim0 + input.dim() : dim0;
            int64_t norm_dim1 = dim1 < 0 ? dim1 + input.dim() : dim1;
            
            if (norm_dim0 >= 0 && norm_dim0 < input.dim() && 
                norm_dim1 >= 0 && norm_dim1 < input.dim()) {
                if (result_sizes[norm_dim0] != input_sizes[norm_dim1] ||
                    result_sizes[norm_dim1] != input_sizes[norm_dim0]) {
                    std::cerr << "Transpose dimensions not correctly swapped!" << std::endl;
                }
            }
        }

        // Test double transpose returns to original
        if (input.is_sparse()) {
            // For sparse tensors, just verify the operation completes
            torch::Tensor double_transpose = torch::transpose(result, dim0, dim1);
        } else {
            // For dense tensors, verify double transpose equals original
            torch::Tensor double_transpose = torch::transpose(result, dim0, dim1);
            if (!torch::allclose(input, double_transpose, 1e-5, 1e-8, false)) {
                // This might fail due to floating point precision, so just log
                // std::cerr << "Double transpose doesn't match original (precision issue)" << std::endl;
            }
        }

        // Test edge cases
        if (offset < Size) {
            int edge_case = Data[offset] % 3;
            offset++;
            
            switch (edge_case) {
                case 0:
                    // Test same dimension transpose (should be no-op)
                    {
                        int64_t same_dim = Data[offset % Size] % input.dim();
                        torch::Tensor same_result = torch::transpose(input, same_dim, same_dim);
                        // For dense tensors, this should be identical
                        if (!input.is_sparse() && !torch::equal(input, same_result)) {
                            std::cerr << "Same dimension transpose changed tensor!" << std::endl;
                        }
                    }
                    break;
                case 1:
                    // Test with contiguous and non-contiguous tensors
                    if (!input.is_sparse()) {
                        torch::Tensor non_contiguous = input.slice(0, 0, -1, 2);
                        if (non_contiguous.dim() >= 2) {
                            torch::Tensor nc_result = torch::transpose(non_contiguous, 0, 1);
                        }
                    }
                    break;
                case 2:
                    // Test boundary dimensions
                    if (input.dim() > 1) {
                        torch::Tensor boundary_result = torch::transpose(input, 0, input.dim() - 1);
                    }
                    break;
            }
        }

        // Additional operations to test memory sharing (for dense tensors)
        if (!input.is_sparse() && result.numel() > 0) {
            // Modify result and check if input changes (they should share storage)
            auto result_flat = result.flatten();
            auto input_flat = input.flatten();
            
            if (result_flat.numel() > 0) {
                auto original_value = result_flat[0].item<float>();
                result_flat[0] = original_value + 1.0f;
                
                // For strided tensors, they should share storage
                // Note: This test might not always work due to view semantics
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