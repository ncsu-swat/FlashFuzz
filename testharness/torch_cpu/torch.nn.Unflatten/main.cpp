#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for unflatten operation
        // Need at least 2 more bytes for dim and sizes
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get dimension to unflatten
        int64_t dim = static_cast<int64_t>(Data[offset++]);
        // Allow negative dimensions for edge case testing
        dim = dim % (2 * input.dim()) - input.dim();
        
        // Get number of dimensions to unflatten into
        uint8_t num_sizes = Data[offset++] % 5 + 1; // 1-5 dimensions
        
        // Parse sizes for the unflattened dimension
        std::vector<int64_t> sizes;
        for (uint8_t i = 0; i < num_sizes && offset < Size; ++i) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t size_val;
                std::memcpy(&size_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow some negative sizes to test error handling
                sizes.push_back(size_val);
            } else {
                // Not enough data, use a small positive value
                sizes.push_back(2);
                offset = Size; // Prevent infinite loop
            }
        }
        
        // If we couldn't parse any sizes, use default values
        if (sizes.empty()) {
            sizes = {2, 2};
        }
        
        // Create the Unflatten module
        torch::nn::Unflatten unflatten(dim, sizes);
        
        // Apply the unflatten operation
        torch::Tensor output = unflatten->forward(input);
        
        // Verify output is not empty
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}