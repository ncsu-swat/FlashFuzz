#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::sqrt

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
            int64_t side_length = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
            
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
            // If tensor has more than 2 dimensions, take the last two dimensions as batch
            // logdet supports batched input of shape (..., n, n)
            std::vector<int64_t> sizes = input.sizes().vec();
            int64_t dim0 = sizes[sizes.size() - 2];
            int64_t dim1 = sizes[sizes.size() - 1];
            
            // Make last two dims square by taking the smaller dimension
            int64_t square_dim = std::min(dim0, dim1);
            if (square_dim < 1) square_dim = 1;
            input = input.narrow(-2, 0, square_dim).narrow(-1, 0, square_dim);
        } else {
            // Already 2D, make it square if needed
            int64_t dim0 = input.size(0);
            int64_t dim1 = input.size(1);
            
            if (dim0 != dim1) {
                int64_t square_dim = std::min(dim0, dim1);
                if (square_dim < 1) square_dim = 1;
                input = input.slice(0, 0, square_dim).slice(1, 0, square_dim);
            }
        }
        
        // Convert to float or double for numerical stability
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat);
        }
        
        // Test 1: Raw input (may be singular, testing edge cases)
        try {
            torch::Tensor result1 = torch::logdet(input);
            // Access result to prevent optimization
            volatile auto dummy1 = result1.data_ptr();
            (void)dummy1;
        } catch (...) {
            // Expected for singular matrices
        }
        
        // Test 2: Make positive-definite for valid logdet computation
        torch::Tensor pd_input;
        if (input.is_complex()) {
            // Make Hermitian by averaging with conjugate transpose
            pd_input = 0.5 * (input + input.transpose(-2, -1).conj());
            
            // Add scaled identity to help make positive-definite
            auto identity = torch::eye(input.size(-1), input.options());
            pd_input = pd_input + identity * (static_cast<float>(input.size(-1)) + 1.0f);
        } else {
            // Make symmetric by averaging with transpose
            pd_input = 0.5 * (input + input.transpose(-2, -1));
            
            // Add scaled identity to help make positive-definite
            auto identity = torch::eye(input.size(-1), input.options());
            pd_input = pd_input + identity * (static_cast<float>(input.size(-1)) + 1.0f);
        }
        
        // Apply logdet operation on positive-definite matrix
        torch::Tensor result = torch::logdet(pd_input);
        
        // Access result to ensure it's not optimized away
        volatile auto dummy = result.data_ptr();
        (void)dummy;
        
        // Test 3: Test with contiguous input
        try {
            torch::Tensor contig_input = pd_input.contiguous();
            torch::Tensor result3 = torch::logdet(contig_input);
            volatile auto dummy3 = result3.data_ptr();
            (void)dummy3;
        } catch (...) {
            // Silently handle
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}