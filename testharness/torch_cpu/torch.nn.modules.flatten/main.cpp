#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Flatten module from the remaining data
        int64_t start_dim = 1;  // Default value
        int64_t end_dim = -1;   // Default value
        
        // If we have more data, use it for start_dim
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_start_dim;
            std::memcpy(&raw_start_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative values for start_dim (PyTorch handles them)
            start_dim = raw_start_dim;
        }
        
        // If we have more data, use it for end_dim
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_end_dim;
            std::memcpy(&raw_end_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative values for end_dim (PyTorch handles them)
            end_dim = raw_end_dim;
        }
        
        // Create and apply the Flatten module using FlattenOptions
        torch::nn::FlattenOptions options(start_dim, end_dim);
        torch::nn::Flatten flatten_module(options);
        torch::Tensor output = flatten_module->forward(input);
        
        // Alternative approach: use the functional API
        torch::Tensor output2 = torch::flatten(input, start_dim, end_dim);
        
        // Ensure the outputs are computed (force evaluation)
        output.sum().item<float>();
        output2.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}