#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for topk from remaining data
        if (offset + 3 >= Size) {
            return 0;
        }
        
        // Skip empty tensors or 0-dim tensors
        if (input.numel() == 0 || input.dim() == 0) {
            return 0;
        }
        
        // Get dimension to perform topk along
        uint8_t dim_byte = Data[offset++];
        int64_t dim = dim_byte % input.dim();
        
        // Get k value (must be <= size along the chosen dimension)
        int64_t dim_size = input.size(dim);
        if (dim_size <= 0) {
            return 0;
        }
        
        uint8_t k_byte = Data[offset++];
        int64_t k = (k_byte % dim_size) + 1; // Ensure k is in [1, dim_size]
        
        // Get largest flag (true = largest values, false = smallest values)
        bool largest = (Data[offset++] % 2) == 0;
        
        // Get sorted flag (true = sort results, false = don't sort)
        bool sorted = (Data[offset++] % 2) == 0;
        
        // Call topk with the main parameter combination
        try {
            auto result = torch::topk(input, k, dim, largest, sorted);
            
            // Access the values and indices to ensure they're computed
            auto values = std::get<0>(result);
            auto indices = std::get<1>(result);
            
            // Use the results to ensure they're valid
            if (values.numel() > 0 && indices.numel() > 0) {
                auto sum = values.sum();
                auto max_idx = indices.max();
                (void)sum;
                (void)max_idx;
            }
        } catch (const c10::Error&) {
            // PyTorch specific exceptions are expected for invalid inputs
        }
        
        // Try with negative dimension (valid in PyTorch)
        if (offset < Size) {
            try {
                int64_t neg_dim = -(1 + (Data[offset++] % input.dim()));
                int64_t neg_dim_size = input.size(neg_dim);
                if (neg_dim_size > 0) {
                    int64_t neg_k = (k_byte % neg_dim_size) + 1;
                    auto result = torch::topk(input, neg_k, neg_dim, largest, sorted);
                    (void)std::get<0>(result);
                }
            } catch (const c10::Error&) {
                // May throw for edge cases
            }
        }
        
        // Try topk with k=1 (edge case)
        try {
            auto result = torch::topk(input, 1, dim, largest, sorted);
            (void)std::get<0>(result);
        } catch (const c10::Error&) {
            // Handle potential errors
        }
        
        // Try topk with k equal to full dimension size
        try {
            auto result = torch::topk(input, dim_size, dim, largest, sorted);
            (void)std::get<0>(result);
        } catch (const c10::Error&) {
            // Handle potential errors
        }
        
        // Try with all combinations of largest and sorted flags
        try {
            auto result1 = torch::topk(input, k, dim, true, true);
            auto result2 = torch::topk(input, k, dim, true, false);
            auto result3 = torch::topk(input, k, dim, false, true);
            auto result4 = torch::topk(input, k, dim, false, false);
            (void)std::get<0>(result1);
            (void)std::get<0>(result2);
            (void)std::get<0>(result3);
            (void)std::get<0>(result4);
        } catch (const c10::Error&) {
            // Handle potential errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}