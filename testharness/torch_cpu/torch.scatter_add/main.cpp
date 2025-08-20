#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
            return 0;
        }
        
        // Create the input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the index tensor (must be long/int64 type for indexing)
        torch::Tensor index;
        if (offset < Size) {
            index = fuzzer_utils::createTensor(Data, Size, offset);
            index = index.to(torch::kInt64);
        } else {
            // If we don't have enough data, create a simple index tensor
            index = torch::tensor({0}, torch::kInt64);
        }
        
        // Create the src tensor to add
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert src to match input's dtype
            src = src.to(input.dtype());
        } else {
            // If we don't have enough data, create a simple src tensor
            src = torch::ones_like(input);
        }
        
        // Get a dimension to scatter along
        int64_t dim = 0;
        if (offset < Size && input.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input.dim());
        }
        
        // Try different variants of scatter_add
        try {
            // Variant 1: Using the scatter_add_ method on the input tensor
            auto result1 = input.clone().scatter_add_(dim, index, src);
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        try {
            // Variant 2: Using the functional form
            auto result2 = torch::scatter_add(input, dim, index, src);
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        // Try with different reduction modes if we have more data
        if (offset < Size) {
            std::string reduction = "sum"; // Default
            uint8_t reduction_selector = Data[offset++];
            if (reduction_selector % 3 == 0) {
                reduction = "sum";
            } else if (reduction_selector % 3 == 1) {
                reduction = "prod";
            } else {
                reduction = "mean";
            }
            
            try {
                // Variant 3: With reduction parameter
                auto result3 = torch::scatter(input, dim, index, src, reduction);
            } catch (const std::exception& e) {
                // Continue
            }
        }
        
        // Try with edge cases if we have more data
        if (offset < Size) {
            // Try with negative dimension
            try {
                int64_t neg_dim = -1;
                if (input.dim() > 0) {
                    neg_dim = -1 * (static_cast<int64_t>(Data[offset++]) % input.dim() + 1);
                }
                auto result4 = torch::scatter_add(input, neg_dim, index, src);
            } catch (const std::exception& e) {
                // Continue
            }
            
            // Try with out-of-bounds indices
            try {
                auto bad_index = index.clone();
                bad_index.fill_(input.size(dim > 0 ? dim : 0) + 1);
                auto result5 = torch::scatter_add(input, dim, bad_index, src);
            } catch (const std::exception& e) {
                // Continue
            }
            
            // Try with mismatched shapes
            try {
                auto mismatched_src = torch::ones({1});
                auto result6 = torch::scatter_add(input, dim, index, mismatched_src);
            } catch (const std::exception& e) {
                // Continue
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