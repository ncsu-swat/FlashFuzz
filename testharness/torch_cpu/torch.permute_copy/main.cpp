#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>
#include <numeric>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        int64_t rank = input_tensor.dim();
        
        // Handle scalar tensor case
        if (rank == 0) {
            // For scalar, permute_copy with empty permutation should work
            torch::Tensor output = torch::permute_copy(input_tensor, {});
            return 0;
        }
        
        // Generate a valid permutation using remaining fuzzer data
        std::vector<int64_t> permutation(rank);
        std::iota(permutation.begin(), permutation.end(), 0); // Fill with 0, 1, 2, ...
        
        // Use fuzzer data to shuffle the permutation
        for (int64_t i = rank - 1; i > 0 && offset < Size; --i) {
            uint8_t swap_byte = Data[offset++];
            int64_t j = swap_byte % (i + 1);
            std::swap(permutation[i], permutation[j]);
        }
        
        // Apply permute_copy operation (this creates a contiguous copy)
        torch::Tensor output = torch::permute_copy(input_tensor, permutation);
        
        // Verify output is contiguous (permute_copy should return contiguous tensor)
        if (!output.is_contiguous()) {
            throw std::runtime_error("permute_copy should return contiguous tensor");
        }
        
        // Verify the output tensor has the expected shape
        std::vector<int64_t> expected_shape;
        for (int64_t dim : permutation) {
            expected_shape.push_back(input_tensor.size(dim));
        }
        
        auto output_sizes = output.sizes().vec();
        if (expected_shape != output_sizes) {
            throw std::runtime_error("Output shape mismatch");
        }
        
        // Verify data integrity for small tensors
        if (input_tensor.numel() > 0 && input_tensor.numel() < 100) {
            // Access first element
            try {
                output.flatten()[0].item<float>();
            } catch (...) {
                // Silently ignore access errors for certain dtypes
            }
        }
        
        // Also test the tensor method version
        torch::Tensor output2 = input_tensor.permute(permutation).contiguous();
        
        // The results should have the same shape
        if (output.sizes() != output2.sizes()) {
            throw std::runtime_error("Shape mismatch between permute_copy and permute");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}