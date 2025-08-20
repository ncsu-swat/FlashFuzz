#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have some data left for generator parameter
        if (offset < Size) {
            // Create a generator or use default
            bool use_generator = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
            
            if (use_generator) {
                // Create a generator
                torch::Generator gen = torch::default_generator;
                
                // Apply poisson operation with generator
                torch::Tensor result = torch::poisson(input, gen);
            } else {
                // Apply poisson operation with default generator
                torch::Tensor result = torch::poisson(input);
            }
        } else {
            // Apply poisson operation with default generator
            torch::Tensor result = torch::poisson(input);
        }
        
        // Try another variant - create a copy and sample from it
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Only attempt if tensor is floating point
            if (input.is_floating_point()) {
                // Create a copy to avoid modifying the original input
                torch::Tensor input_copy = input.clone();
                
                // Apply poisson sampling with inplace operation
                input_copy.poisson_();
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