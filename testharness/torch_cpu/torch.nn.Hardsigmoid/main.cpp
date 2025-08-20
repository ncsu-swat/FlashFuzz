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
        
        // Apply Hardsigmoid to the input tensor using functional interface
        torch::Tensor output = torch::hardsigmoid(input);
        
        // Try inplace version if there's enough data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            torch::hardsigmoid_(input_copy);
        }
        
        // Try with functional interface
        torch::Tensor output3 = torch::hardsigmoid(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}