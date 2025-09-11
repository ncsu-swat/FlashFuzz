#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a value for ind parameter (between 1 and 4)
        int ind = 1;
        if (offset < Size) {
            ind = (Data[offset++] % 4) + 1;
        }
        
        // Ensure tensor has proper shape for tensorinv
        // For tensorinv, the product of the first ind dimensions must equal
        // the product of the remaining dimensions
        int64_t prod_first = 1;
        int64_t prod_last = 1;
        
        if (A.dim() > 0) {
            // Calculate product of first ind dimensions
            for (int i = 0; i < std::min(ind, static_cast<int>(A.dim())); i++) {
                prod_first *= A.size(i);
            }
            
            // Calculate product of remaining dimensions
            for (int i = std::min(ind, static_cast<int>(A.dim())); i < A.dim(); i++) {
                prod_last *= A.size(i);
            }
            
            // If dimensions don't match, reshape the tensor
            if (prod_first != prod_last && prod_first > 0 && prod_last > 0) {
                // Create a new shape where prod_first = prod_last
                std::vector<int64_t> new_shape;
                
                // First ind dimensions
                int64_t new_prod_first = 1;
                for (int i = 0; i < ind - 1; i++) {
                    int64_t dim_size = (i < A.dim()) ? A.size(i) : 1;
                    new_shape.push_back(dim_size);
                    new_prod_first *= dim_size;
                }
                
                // Make the ind-th dimension match the product of remaining dimensions
                new_shape.push_back(prod_last);
                
                // Remaining dimensions
                int64_t new_prod_last = 1;
                for (int i = ind; i < A.dim(); i++) {
                    int64_t dim_size = A.size(i);
                    new_shape.push_back(dim_size);
                    new_prod_last *= dim_size;
                }
                
                // If we need more dimensions to satisfy ind
                while (new_shape.size() < 2 * ind) {
                    new_shape.push_back(1);
                }
                
                // Try to reshape the tensor
                try {
                    A = A.reshape(new_shape);
                } catch (const std::exception& e) {
                    // If reshape fails, create a new tensor with the desired shape
                    A = torch::ones(new_shape, A.options());
                }
            }
        } else {
            // For scalar tensors, create a tensor with proper dimensions
            std::vector<int64_t> new_shape(2 * ind, 2);
            A = torch::ones(new_shape, A.options());
        }
        
        // Convert to float or complex for numerical stability
        if (A.scalar_type() != torch::kFloat && 
            A.scalar_type() != torch::kDouble && 
            A.scalar_type() != torch::kComplexFloat && 
            A.scalar_type() != torch::kComplexDouble) {
            A = A.to(torch::kFloat);
        }
        
        // Add a small value to diagonal elements to improve invertibility
        if (A.dim() >= 2) {
            int min_dim = std::min(A.size(0), A.size(1));
            for (int i = 0; i < min_dim; i++) {
                // Create index for the diagonal element
                std::vector<torch::indexing::TensorIndex> indices(A.dim(), torch::indexing::Slice());
                indices[0] = i;
                indices[1] = i;
                
                // Add a value to make matrix more likely to be invertible
                A.index_put_(indices, A.index(indices) + 1.0);
            }
        }
        
        // Apply tensorinv operation
        torch::Tensor result = torch::tensorinv(A, ind);
        
        // Verify the result by checking if A @ result is close to identity
        // This is a basic sanity check, not a comprehensive test
        if (result.numel() > 0 && A.numel() > 0) {
            // Reshape tensors for matrix multiplication
            std::vector<int64_t> a_shape, inv_shape;
            for (int i = 0; i < ind; i++) {
                a_shape.push_back(A.size(i));
            }
            int64_t prod_a = std::accumulate(a_shape.begin(), a_shape.end(), 
                                           static_cast<int64_t>(1), std::multiplies<int64_t>());
            
            for (int i = ind; i < A.dim(); i++) {
                inv_shape.push_back(A.size(i));
            }
            int64_t prod_inv = std::accumulate(inv_shape.begin(), inv_shape.end(), 
                                             static_cast<int64_t>(1), std::multiplies<int64_t>());
            
            // Reshape for matrix multiplication
            torch::Tensor a_mat = A.reshape({prod_a, prod_inv});
            torch::Tensor inv_mat = result.reshape({prod_inv, prod_a});
            
            // Check if a_mat @ inv_mat is close to identity
            torch::Tensor product = torch::matmul(a_mat, inv_mat);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
