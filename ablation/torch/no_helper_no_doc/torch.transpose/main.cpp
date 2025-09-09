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

        // Extract tensor dimensions (1-6D tensors)
        uint8_t num_dims = (Data[offset++] % 6) + 1; // 1 to 6 dimensions
        
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 10) + 1; // 1 to 10 elements per dimension
            shape.push_back(dim_size);
        }

        if (offset + 2 >= Size) {
            return 0;
        }

        // Extract dimension indices for transpose
        int64_t dim0 = static_cast<int64_t>(Data[offset++] % num_dims);
        int64_t dim1 = static_cast<int64_t>(Data[offset++] % num_dims);

        // Test with negative indices as well
        if (offset < Size && Data[offset++] % 2 == 0) {
            dim0 = dim0 - num_dims; // Convert to negative index
        }
        if (offset < Size && Data[offset++] % 2 == 0) {
            dim1 = dim1 - num_dims; // Convert to negative index
        }

        // Create tensor with various data types
        torch::Tensor tensor;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 6;
            switch (dtype_choice) {
                case 0:
                    tensor = torch::randn(shape, torch::kFloat32);
                    break;
                case 1:
                    tensor = torch::randn(shape, torch::kFloat64);
                    break;
                case 2:
                    tensor = torch::randint(0, 100, shape, torch::kInt32);
                    break;
                case 3:
                    tensor = torch::randint(0, 100, shape, torch::kInt64);
                    break;
                case 4:
                    tensor = torch::randint(0, 2, shape, torch::kBool);
                    break;
                default:
                    tensor = torch::randn(shape, torch::kFloat16);
                    break;
            }
        } else {
            tensor = torch::randn(shape);
        }

        // Test torch::transpose function
        torch::Tensor result = torch::transpose(tensor, dim0, dim1);

        // Verify the result has expected properties
        auto original_shape = tensor.sizes();
        auto result_shape = result.sizes();
        
        // Check that dimensions are swapped correctly
        if (result_shape.size() == original_shape.size()) {
            // Normalize negative indices for verification
            int64_t norm_dim0 = dim0 < 0 ? dim0 + num_dims : dim0;
            int64_t norm_dim1 = dim1 < 0 ? dim1 + num_dims : dim1;
            
            if (norm_dim0 >= 0 && norm_dim0 < num_dims && 
                norm_dim1 >= 0 && norm_dim1 < num_dims) {
                // Verify shape is correct after transpose
                for (int64_t i = 0; i < num_dims; i++) {
                    int64_t expected_size;
                    if (i == norm_dim0) {
                        expected_size = original_shape[norm_dim1];
                    } else if (i == norm_dim1) {
                        expected_size = original_shape[norm_dim0];
                    } else {
                        expected_size = original_shape[i];
                    }
                    
                    if (result_shape[i] != expected_size) {
                        std::cout << "Shape mismatch after transpose" << std::endl;
                    }
                }
            }
        }

        // Test tensor.transpose() method as well
        torch::Tensor result2 = tensor.transpose(dim0, dim1);

        // Test transpose with same dimension (should be no-op)
        if (offset < Size && Data[offset++] % 4 == 0) {
            torch::Tensor same_dim_result = torch::transpose(tensor, dim0, dim0);
            // Should have same shape as original
            if (!same_dim_result.sizes().equals(tensor.sizes())) {
                std::cout << "Same dimension transpose changed shape" << std::endl;
            }
        }

        // Test chained transposes
        if (offset < Size && Data[offset++] % 3 == 0 && num_dims >= 3) {
            int64_t dim2 = static_cast<int64_t>(Data[offset % Size] % num_dims);
            torch::Tensor chained = torch::transpose(torch::transpose(tensor, dim0, dim1), dim1, dim2);
        }

        // Test with requires_grad tensor
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kFloat64) {
                tensor.requires_grad_(true);
                torch::Tensor grad_result = torch::transpose(tensor, dim0, dim1);
                
                // Test backward pass
                if (grad_result.numel() > 0) {
                    torch::Tensor loss = grad_result.sum();
                    loss.backward();
                }
            }
        }

        // Force evaluation to catch any lazy evaluation issues
        result.sum().item<double>();
        result2.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}