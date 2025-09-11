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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Flatten module from the remaining data
        int64_t start_dim = 1;  // Default value
        int64_t end_dim = -1;   // Default value
        
        // If we have more data, use it to set start_dim
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_start_dim;
            std::memcpy(&raw_start_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative values for start_dim (PyTorch handles them)
            start_dim = raw_start_dim;
        }
        
        // If we have more data, use it to set end_dim
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_end_dim;
            std::memcpy(&raw_end_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative values for end_dim (PyTorch handles them)
            end_dim = raw_end_dim;
        }
        
        // Create the Flatten module with options
        torch::nn::FlattenOptions options(start_dim, end_dim);
        torch::nn::Flatten flatten_module(options);
        
        // Apply the Flatten operation
        torch::Tensor output = flatten_module->forward(input);
        
        // Ensure the output is valid by accessing some property
        auto output_sizes = output.sizes();
        
        // Alternative approach: use the functional API
        torch::Tensor output2 = torch::flatten(input, start_dim, end_dim);
        
        // Check if both approaches give the same result
        if (output.defined() && output2.defined()) {
            bool shapes_match = output.sizes() == output2.sizes();
            bool values_match = torch::all(torch::eq(output, output2)).item<bool>();
            
            // If there's a mismatch, we might want to investigate
            if (!shapes_match || !values_match) {
                return 1; // Keep the input for investigation
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
