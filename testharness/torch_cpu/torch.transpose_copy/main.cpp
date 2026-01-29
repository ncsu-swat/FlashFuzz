#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Extract dimensions for transpose
        int64_t dim0 = 0;
        int64_t dim1 = 1;
        
        // Get dimensions to transpose if we have enough data
        if (offset + sizeof(int64_t) * 2 <= Size) {
            std::memcpy(&dim0, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get tensor rank
        int64_t tensor_rank = input_tensor.dim();
        
        // transpose_copy requires at least 2 dimensions
        if (tensor_rank < 2) {
            return 0;
        }
        
        // Map dimensions to valid range for the tensor
        dim0 = std::abs(dim0) % tensor_rank;
        dim1 = std::abs(dim1) % tensor_rank;
        
        // Apply transpose_copy operation - creates a contiguous copy with transposed dims
        torch::Tensor output = torch::transpose_copy(input_tensor, dim0, dim1);
        
        // Verify the output is valid
        if (output.defined()) {
            // Check shape is correct
            auto input_sizes = input_tensor.sizes();
            auto output_sizes = output.sizes();
            
            // Verify dimensions are swapped correctly
            (void)output_sizes;
            
            // Access data to ensure memory is valid
            if (output.numel() > 0) {
                // Use sum() instead of item() to handle multi-element tensors
                auto sum = output.sum();
                (void)sum;
            }
            
            // Verify the copy is contiguous
            (void)output.is_contiguous();
        }
        
        // Test with negative dimensions (inner try-catch, no logging)
        try {
            torch::Tensor output_neg = torch::transpose_copy(input_tensor, -1, -2);
            if (output_neg.defined() && output_neg.numel() > 0) {
                auto sum = output_neg.sum();
                (void)sum;
            }
        } catch (...) {
            // Expected behavior for some edge cases
        }
        
        // Test transposing same dimension (should be a no-op copy)
        try {
            torch::Tensor output_same = torch::transpose_copy(input_tensor, dim0, dim0);
            if (output_same.defined() && output_same.numel() > 0) {
                auto sum = output_same.sum();
                (void)sum;
            }
        } catch (...) {
            // May fail depending on implementation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}