#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic parameters
        if (Size < 10) return 0;

        // Extract fuzzing parameters
        bool sorted = extractBool(Data, Size, offset);
        bool return_inverse = extractBool(Data, Size, offset);
        bool return_counts = extractBool(Data, Size, offset);
        bool use_dim = extractBool(Data, Size, offset);
        
        // Extract tensor parameters
        auto dtype = extractDtype(Data, Size, offset);
        auto shape = extractShape(Data, Size, offset, 1, 4); // 1-4 dimensions
        
        // Create input tensor with extracted shape and dtype
        torch::Tensor input = createTensor(Data, Size, offset, shape, dtype);
        
        // Test case 1: Basic unique without dim
        if (!use_dim) {
            if (!return_inverse && !return_counts) {
                // Just unique values
                auto result = torch::unique(input, sorted);
                // Verify result is valid
                if (result.numel() > input.numel()) {
                    throw std::runtime_error("Unique result has more elements than input");
                }
            } else if (return_inverse && !return_counts) {
                // Unique with inverse indices
                auto [unique_vals, inverse_indices] = torch::unique(input, sorted, return_inverse);
                // Verify inverse indices shape matches input
                if (!inverse_indices.sizes().equals(input.sizes())) {
                    throw std::runtime_error("Inverse indices shape mismatch");
                }
            } else if (!return_inverse && return_counts) {
                // Unique with counts
                auto [unique_vals, counts] = torch::unique(input, sorted, return_inverse, return_counts);
                // Verify counts shape matches unique values
                if (counts.numel() != unique_vals.numel()) {
                    throw std::runtime_error("Counts shape mismatch with unique values");
                }
            } else {
                // All return options
                auto [unique_vals, inverse_indices, counts] = torch::unique(input, sorted, return_inverse, return_counts);
                // Verify shapes
                if (!inverse_indices.sizes().equals(input.sizes())) {
                    throw std::runtime_error("Inverse indices shape mismatch");
                }
                if (counts.numel() != unique_vals.numel()) {
                    throw std::runtime_error("Counts shape mismatch with unique values");
                }
            }
        } else {
            // Test case 2: Unique along a specific dimension
            if (input.dim() == 0) return 0; // Skip scalar tensors for dim operation
            
            int64_t dim = extractInt(Data, Size, offset) % input.dim();
            
            if (!return_inverse && !return_counts) {
                auto result = torch::unique(input, sorted, return_inverse, return_counts, dim);
                // Verify result has same number of dimensions
                if (result.dim() != input.dim()) {
                    throw std::runtime_error("Unique result dimension mismatch");
                }
            } else if (return_inverse && !return_counts) {
                auto [unique_vals, inverse_indices] = torch::unique(input, sorted, return_inverse, return_counts, dim);
                if (unique_vals.dim() != input.dim()) {
                    throw std::runtime_error("Unique result dimension mismatch");
                }
                // Inverse indices should have shape matching the specified dimension
                if (inverse_indices.size(dim) != input.size(dim)) {
                    throw std::runtime_error("Inverse indices dimension size mismatch");
                }
            } else if (!return_inverse && return_counts) {
                auto [unique_vals, counts] = torch::unique(input, sorted, return_inverse, return_counts, dim);
                if (unique_vals.dim() != input.dim()) {
                    throw std::runtime_error("Unique result dimension mismatch");
                }
                if (counts.numel() != unique_vals.size(dim)) {
                    throw std::runtime_error("Counts size mismatch");
                }
            } else {
                auto [unique_vals, inverse_indices, counts] = torch::unique(input, sorted, return_inverse, return_counts, dim);
                if (unique_vals.dim() != input.dim()) {
                    throw std::runtime_error("Unique result dimension mismatch");
                }
                if (inverse_indices.size(dim) != input.size(dim)) {
                    throw std::runtime_error("Inverse indices dimension size mismatch");
                }
                if (counts.numel() != unique_vals.size(dim)) {
                    throw std::runtime_error("Counts size mismatch");
                }
            }
        }

        // Test edge cases with different tensor configurations
        if (offset < Size) {
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0}, dtype);
            auto empty_result = torch::unique(empty_tensor, sorted, return_inverse, return_counts);
            
            // Test with single element tensor
            torch::Tensor single_elem = torch::ones({1}, dtype);
            auto single_result = torch::unique(single_elem, sorted, return_inverse, return_counts);
            
            // Test with all same elements
            if (input.numel() > 1) {
                torch::Tensor same_elements = torch::full_like(input, input.flatten()[0]);
                auto same_result = torch::unique(same_elements, sorted, return_inverse, return_counts);
            }
        }

        // Additional stress test: large tensor if we have enough data
        if (Size > 1000 && offset < Size - 100) {
            auto large_shape = extractShape(Data, Size, offset, 2, 3); // 2-3 dimensions
            // Limit size to prevent excessive memory usage
            int64_t total_elements = 1;
            for (auto s : large_shape) {
                total_elements *= s;
                if (total_elements > 10000) break;
            }
            if (total_elements <= 10000) {
                torch::Tensor large_input = createTensor(Data, Size, offset, large_shape, dtype);
                auto large_result = torch::unique(large_input, sorted);
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