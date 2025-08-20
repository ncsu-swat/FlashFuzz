#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make the tensor square and symmetric to satisfy Cholesky requirements
        if (input.dim() >= 2) {
            // Get the minimum of the last two dimensions to make it square
            int64_t min_dim = std::min(input.size(-1), input.size(-2));
            
            // Slice to make square
            if (input.dim() == 2) {
                input = input.slice(0, 0, min_dim).slice(1, 0, min_dim);
            } else {
                // For higher dimensions, only make the last two dimensions square
                input = input.slice(-2, 0, min_dim).slice(-1, 0, min_dim);
            }
            
            // Make symmetric: A = 0.5 * (A + A.transpose(-2, -1))
            input = 0.5 * (input + input.transpose(-2, -1));
            
            // Make positive definite by adding a diagonal matrix
            // Get the number of dimensions
            int64_t ndim = input.dim();
            
            // Create an identity tensor with the same batch dimensions
            std::vector<int64_t> identity_size(input.sizes().begin(), input.sizes().end());
            torch::Tensor identity = torch::eye(min_dim, input.options());
            
            // Broadcast identity to match input's batch dimensions if needed
            if (ndim > 2) {
                std::vector<int64_t> expanded_size(ndim, 1);
                expanded_size[ndim-2] = min_dim;
                expanded_size[ndim-1] = min_dim;
                identity = identity.view(expanded_size);
                
                std::vector<int64_t> expand_size(input.sizes().begin(), input.sizes().end());
                identity = identity.expand(expand_size);
            }
            
            // Add identity matrix to ensure positive definiteness
            // Using a large enough value to ensure numerical stability
            input = input + min_dim * identity;
        }
        
        // Apply Cholesky decomposition
        try {
            torch::Tensor result = torch::cholesky(input);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
        
        // Try with upper triangular option if we have more data
        if (offset < Size) {
            bool upper = Data[offset++] % 2 == 1;
            
            try {
                torch::Tensor result = torch::cholesky(input, upper);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
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