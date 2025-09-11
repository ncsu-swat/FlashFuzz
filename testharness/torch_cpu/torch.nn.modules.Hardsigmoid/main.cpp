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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Hardsigmoid module
        torch::nn::Hardsigmoid hardsigmoid_module;
        
        // Apply Hardsigmoid to the input tensor
        torch::Tensor output = hardsigmoid_module(input);
        
        // Try with inplace version if there's enough data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            
            // Set inplace option based on next byte
            bool inplace = Data[offset++] % 2 == 1;
            
            // Apply functional version
            torch::Tensor output_func;
            if (inplace) {
                output_func = torch::hardsigmoid_(input_copy);
            } else {
                output_func = torch::hardsigmoid(input_copy);
            }
        }
        
        // Try with different configurations if there's more data
        if (offset + 1 < Size) {
            // Create a new tensor
            torch::Tensor another_input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try with different data types
            if (another_input.dtype() != torch::kFloat) {
                // Convert to float for numerical stability
                another_input = another_input.to(torch::kFloat);
            }
            
            // Apply Hardsigmoid
            torch::Tensor another_output = hardsigmoid_module(another_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
