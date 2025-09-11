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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a square matrix for logdet
        // logdet requires a square matrix (n x n)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // logdet requires a square matrix, so we need to ensure it's 2D and square
        if (input.dim() < 2) {
            // If tensor is 0D or 1D, reshape it to a square matrix
            int64_t total_elements = input.numel();
            int64_t side_length = static_cast<int64_t>(std::sqrt(total_elements));
            
            // Ensure we have at least a 1x1 matrix
            side_length = std::max(side_length, static_cast<int64_t>(1));
            
            // Reshape to square matrix, padding with zeros if needed
            torch::Tensor square_input;
            if (side_length * side_length != total_elements) {
                // Need to create a new tensor of the right size
                square_input = torch::zeros({side_length, side_length}, input.options());
                // Copy as many elements as possible from the original tensor
                int64_t elements_to_copy = std::min(total_elements, side_length * side_length);
                if (elements_to_copy > 0) {
                    auto flat_input = input.reshape({-1});
                    auto flat_square = square_input.reshape({-1});
                    flat_square.slice(0, 0, elements_to_copy).copy_(flat_input.slice(0, 0, elements_to_copy));
                }
            } else {
                // Perfect square, just reshape
                square_input = input.reshape({side_length, side_length});
            }
            input = square_input;
        } else if (input.dim() > 2) {
            // If tensor has more than 2 dimensions, take the first 2D slice
            std::vector<int64_t> sizes = input.sizes().vec();
            int64_t dim0 = sizes[0];
            int64_t dim1 = sizes[1];
            
            // Make it square by taking the smaller dimension
            int64_t square_dim = std::min(dim0, dim1);
            input = input.slice(0, 0, square_dim).slice(1, 0, square_dim);
        } else {
            // Already 2D, make it square if needed
            int64_t dim0 = input.size(0);
            int64_t dim1 = input.size(1);
            
            if (dim0 != dim1) {
                int64_t square_dim = std::min(dim0, dim1);
                input = input.slice(0, 0, square_dim).slice(1, 0, square_dim);
            }
        }
        
        // For complex tensors, ensure they're Hermitian positive-definite
        // For real tensors, ensure they're symmetric positive-definite
        // This is not strictly necessary for the fuzzer, but helps test more valid inputs
        if (input.is_complex()) {
            // Make Hermitian by averaging with conjugate transpose
            input = 0.5 * (input + input.transpose(-2, -1).conj());
            
            // Add to identity to help make positive-definite
            auto identity = torch::eye(input.size(0), input.options());
            input = input + identity * (input.size(0) + 1);
        } else {
            // Make symmetric by averaging with transpose
            input = 0.5 * (input + input.transpose(-2, -1));
            
            // Add to identity to help make positive-definite
            auto identity = torch::eye(input.size(0), input.options());
            input = input + identity * (input.size(0) + 1);
        }
        
        // Convert to float or double for numerical stability
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat);
        }
        
        // Apply logdet operation
        torch::Tensor result = torch::logdet(input);
        
        // Optional: Try to use the result to ensure it's not optimized away
        if (result.numel() > 0) {
            volatile float dummy = result.item<float>();
            (void)dummy;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
