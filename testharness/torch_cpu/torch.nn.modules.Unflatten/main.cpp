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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Unflatten module
        // We need at least 2 more bytes for dim and sizes
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get dimension to unflatten
        int64_t dim = static_cast<int64_t>(Data[offset++]);
        // Allow negative dimensions for edge case testing
        dim = dim % 10 - 5;  // Range: -5 to 4
        
        // Get number of sizes for unflattened dimension
        uint8_t num_sizes = Data[offset++] % 5 + 1;  // Range: 1 to 5
        
        // Parse sizes for unflattened dimension
        std::vector<int64_t> sizes;
        for (uint8_t i = 0; i < num_sizes && offset < Size; ++i) {
            if (offset < Size) {
                int64_t size_val = static_cast<int64_t>(Data[offset++]) % 8 + 1;  // Range: 1 to 8
                sizes.push_back(size_val);
            }
        }
        
        // Create Unflatten module
        torch::nn::Unflatten unflatten_module(dim, sizes);
        
        // Apply the unflatten operation
        torch::Tensor output = unflatten_module->forward(input_tensor);
        
        // Try another variant with named sizes
        if (offset + 1 < Size) {
            // Get a different dimension name
            std::string dim_name = "batch";
            
            // Create a named sizes map
            torch::nn::UnflattenOptions::namedshape_t named_sizes;
            uint8_t num_named_sizes = Data[offset++] % 3 + 1;  // Range: 1 to 3
            
            std::vector<std::string> dimension_names = {"height", "width", "depth", "channels"};
            
            for (uint8_t i = 0; i < num_named_sizes && offset < Size; ++i) {
                if (offset < Size) {
                    std::string name = dimension_names[i % dimension_names.size()];
                    int64_t size_val = static_cast<int64_t>(Data[offset++]) % 8 + 1;  // Range: 1 to 8
                    named_sizes[name] = size_val;
                }
            }
            
            // Create Unflatten module with named sizes
            torch::nn::Unflatten unflatten_named(dim_name, named_sizes);
            
            // Apply the unflatten operation with named sizes
            torch::Tensor output_named = unflatten_named->forward(input_tensor);
        }
        
        // Try edge case: empty sizes vector
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::nn::Unflatten unflatten_empty(0, std::vector<int64_t>{});
                torch::Tensor output_empty = unflatten_empty->forward(input_tensor);
            } catch (...) {
                // Expected to fail, but let's catch it to continue testing
            }
        }
        
        // Try edge case: sizes that don't match the tensor
        if (offset < Size) {
            int64_t dim3 = static_cast<int64_t>(Data[offset++]) % 10 - 5;
            std::vector<int64_t> incompatible_sizes = {100, 100, 100};
            try {
                torch::nn::Unflatten unflatten_incompatible(dim3, incompatible_sizes);
                torch::Tensor output_incompatible = unflatten_incompatible->forward(input_tensor);
            } catch (...) {
                // Expected to fail in many cases, but let's catch it to continue testing
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