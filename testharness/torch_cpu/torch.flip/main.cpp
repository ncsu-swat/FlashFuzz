#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Handle 0-dim tensors - flip requires at least 1 dimension
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Parse dimensions to flip
        std::vector<int64_t> dims_to_flip;
        
        // Determine number of dimensions to flip (at least 1, at most tensor.dim())
        if (offset < Size) {
            uint8_t num_dims_to_flip = 1 + (Data[offset++] % input_tensor.dim());
            
            // Parse each dimension to flip
            for (uint8_t i = 0; i < num_dims_to_flip && offset < Size; ++i) {
                // Map byte to valid dimension range [-dim, dim-1]
                int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                int64_t dim = dim_byte % input_tensor.dim();
                dims_to_flip.push_back(dim);
            }
        }
        
        // If no dimensions were specified, flip dimension 0
        if (dims_to_flip.empty()) {
            dims_to_flip.push_back(0);
        }
        
        // Apply torch.flip operation
        torch::Tensor output;
        
        // Inner try-catch for expected failures (invalid dimensions, etc.)
        try {
            output = torch::flip(input_tensor, dims_to_flip);
        } catch (const std::exception &) {
            // Expected failure for invalid dimension combinations
            return 0;
        }
        
        // Verify output shape matches input shape
        if (output.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("Output tensor has different shape than input tensor");
        }
        
        // Test flipping with single dimensions
        for (int64_t d = 0; d < input_tensor.dim(); ++d) {
            try {
                torch::Tensor single_flip = torch::flip(input_tensor, {d});
                // Also test negative indexing
                torch::Tensor neg_flip = torch::flip(input_tensor, {-d - 1});
            } catch (const std::exception &) {
                // Silently ignore expected failures
            }
        }
        
        // Test double flip (should restore original)
        try {
            torch::Tensor double_flip = torch::flip(output, dims_to_flip);
            // Result should be close to original
        } catch (const std::exception &) {
            // Silently ignore
        }
        
        // Test flipping all dimensions
        try {
            std::vector<int64_t> all_dims;
            for (int64_t d = 0; d < input_tensor.dim(); ++d) {
                all_dims.push_back(d);
            }
            torch::Tensor all_flip = torch::flip(input_tensor, all_dims);
        } catch (const std::exception &) {
            // Silently ignore
        }
        
        // Test with empty dims vector (edge case)
        try {
            torch::Tensor no_flip = torch::flip(input_tensor, {});
        } catch (const std::exception &) {
            // Expected - empty dims may or may not be allowed
        }
        
        // Basic operations on output to ensure tensor validity
        try {
            torch::Tensor sum = output.sum();
            torch::Tensor contiguous = output.contiguous();
        } catch (const std::exception &) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}