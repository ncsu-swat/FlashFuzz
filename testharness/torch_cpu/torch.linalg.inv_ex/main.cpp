#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure we have a square matrix for inv_ex
        // If tensor is 0D or 1D, reshape to 2x2
        if (A.dim() < 2) {
            A = A.reshape({2, 2});
        } else {
            // For tensors with dim >= 2, make the last two dimensions equal
            // to ensure a square matrix for each batch element
            auto sizes = A.sizes().vec();
            int64_t square_dim = std::max(sizes[sizes.size() - 1], sizes[sizes.size() - 2]);
            sizes[sizes.size() - 1] = square_dim;
            sizes[sizes.size() - 2] = square_dim;
            A = A.reshape(sizes);
        }
        
        // Try to make the tensor invertible by adding a scaled identity matrix
        // This helps explore more code paths rather than always failing with singular matrices
        if (offset + 1 < Size) {
            uint8_t scale_factor = Data[offset++];
            auto identity = torch::eye(A.size(-1), A.options());
            
            // Expand identity to match A's batch dimensions if needed
            if (A.dim() > 2) {
                std::vector<int64_t> expanded_sizes(A.dim(), 1);
                expanded_sizes[A.dim() - 2] = A.size(-2);
                expanded_sizes[A.dim() - 1] = A.size(-1);
                identity = identity.expand(A.sizes());
            }
            
            // Add scaled identity to improve conditioning
            A = A + (scale_factor / 10.0) * identity;
        }
        
        // Apply torch.linalg.inv_ex operation
        auto result = torch::inverse(A);
        
        // Use the result to prevent optimization
        auto check_sum = result.sum();
        if (check_sum.item<double>() == -12345.6789) {
            // This will never happen, just to prevent compiler from optimizing away
            std::cerr << "Unreachable" << std::endl;
        }
        
        // Test with different dtypes if we have more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            dtype_selector = dtype_selector % 2; // Just use 2 different dtypes
            
            torch::ScalarType target_dtype;
            if (dtype_selector == 0) {
                target_dtype = torch::kFloat;
            } else {
                target_dtype = torch::kDouble;
            }
            
            // Convert tensor to the selected dtype
            auto A_converted = A.to(target_dtype);
            
            // Apply inverse on the converted tensor
            auto result2 = torch::inverse(A_converted);
            
            // Use the results
            auto check_sum2 = result2.sum();
            if (check_sum2.item<double>() == -12345.6789) {
                std::cerr << "Unreachable" << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}