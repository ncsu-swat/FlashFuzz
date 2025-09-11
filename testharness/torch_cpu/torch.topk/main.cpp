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
        
        // Get k value (number of top elements to return)
        int64_t k = 1;
        if (!input.numel()) {
            // If tensor is empty, use default k=1
            k = 1;
        } else {
            // Get a dimension to use for k
            int64_t max_k = input.numel();
            if (max_k > 0) {
                // Use the next byte to determine k
                uint8_t k_byte = Data[offset++];
                k = (k_byte % max_k) + 1; // Ensure k is at least 1 and at most numel
            }
        }
        
        // Get dimension to perform topk along
        int64_t dim = 0;
        if (input.dim() > 0) {
            uint8_t dim_byte = Data[offset++];
            dim = dim_byte % input.dim(); // Ensure dim is valid
        }
        
        // Get largest flag (true = largest values, false = smallest values)
        bool largest = (Data[offset++] % 2) == 0;
        
        // Get sorted flag (true = sort results, false = don't sort)
        bool sorted = (Data[offset++] % 2) == 0;
        
        // Call topk with different parameter combinations
        try {
            auto result = torch::topk(input, k, dim, largest, sorted);
            
            // Access the values and indices to ensure they're computed
            auto values = std::get<0>(result);
            auto indices = std::get<1>(result);
            
            // Try to use the results to ensure they're valid
            if (values.numel() > 0 && indices.numel() > 0) {
                auto sum = values.sum();
                auto max_idx = indices.max();
            }
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected for invalid inputs
            return 0;
        }
        
        // Try with different k values
        if (offset < Size && input.numel() > 0) {
            uint8_t alt_k_byte = Data[offset++];
            // Try with k=0 (should throw an exception)
            try {
                auto result = torch::topk(input, 0, dim, largest, sorted);
            } catch (const c10::Error&) {
                // Expected exception for k=0
            }
            
            // Try with k > dimension size (should be clamped or throw)
            try {
                int64_t dim_size = 1;
                if (input.dim() > 0) {
                    dim_size = input.size(dim);
                }
                
                if (dim_size > 0) {
                    int64_t large_k = dim_size + (alt_k_byte % 10) + 1;
                    auto result = torch::topk(input, large_k, dim, largest, sorted);
                }
            } catch (const c10::Error&) {
                // Expected exception for k > dimension size
            }
        }
        
        // Try with negative dimension
        if (offset < Size && input.dim() > 0) {
            try {
                int64_t neg_dim = -1 - (Data[offset++] % input.dim());
                auto result = torch::topk(input, k, neg_dim, largest, sorted);
            } catch (const c10::Error&) {
                // May throw for invalid negative dimension
            }
        }
        
        // Try with out-of-bounds dimension
        if (offset < Size) {
            try {
                int64_t invalid_dim = input.dim() + (Data[offset++] % 5);
                auto result = torch::topk(input, k, invalid_dim, largest, sorted);
            } catch (const c10::Error&) {
                // Expected exception for invalid dimension
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
