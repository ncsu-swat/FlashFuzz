#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Flatten module
        int64_t start_dim = 1;  // Default value
        int64_t end_dim = -1;   // Default value
        
        // Get start_dim from input data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get end_dim from input data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&end_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create Flatten module using FlattenOptions
        torch::nn::FlattenOptions options(start_dim, end_dim);
        torch::nn::Flatten flatten_module(options);
        
        // Apply the Flatten operation
        torch::Tensor output = flatten_module->forward(input);
        
        // Alternative way to test: use the functional API
        torch::Tensor output2 = torch::flatten(input, start_dim, end_dim);
        
        // Test edge case: create another Flatten with different parameters
        int64_t alt_start_dim = 0;
        int64_t alt_end_dim = 1;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&alt_start_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&alt_end_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try with alternative parameters
        try {
            torch::nn::FlattenOptions alt_options(alt_start_dim, alt_end_dim);
            torch::nn::Flatten alt_flatten(alt_options);
            torch::Tensor alt_output = alt_flatten->forward(input);
        } catch (const std::exception&) {
            // Expected to potentially throw for invalid parameters
        }
        
        // Test with empty tensor if input has at least one dimension
        if (input.dim() > 0) {
            try {
                auto empty_tensor = torch::empty({0}, input.options());
                torch::Tensor empty_output = flatten_module->forward(empty_tensor);
            } catch (const std::exception&) {
                // Expected to potentially throw for empty tensor
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