#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create GELU module with different approximation types
        uint8_t approx_type_byte = (offset < Size) ? Data[offset++] : 0;
        std::string approximation = "none";
        
        // Select approximation type based on input data
        if (approx_type_byte % 3 == 1) {
            approximation = "tanh";
        } else if (approx_type_byte % 3 == 2) {
            approximation = "none";
        }
        
        // Create GELU module with options
        torch::nn::GELUOptions options;
        options.approximate(approximation);
        torch::nn::GELU gelu_module(options);
        
        // Apply GELU operation
        torch::Tensor output = gelu_module->forward(input);
        
        // Try the functional version as well
        torch::Tensor output_functional = torch::gelu(input, approximation);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}