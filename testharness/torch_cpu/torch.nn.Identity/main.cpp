#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Identity module with default configuration
        torch::nn::Identity identity_default;
        
        // Apply Identity operation
        torch::Tensor output_tensor = identity_default->forward(input_tensor);
        
        // Try with sequential module
        torch::nn::Sequential sequential(
            (torch::nn::Identity())
        );
        torch::Tensor output_sequential = sequential->forward(input_tensor);
        
        // Try with functional API - just return the input tensor as identity
        torch::Tensor output_functional = input_tensor.clone();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}