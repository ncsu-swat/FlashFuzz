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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dim parameter from the remaining data
        int64_t dim = 1;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create GLU module
        torch::nn::GLU glu_module(torch::nn::GLUOptions().dim(dim));
        
        // Apply GLU operation
        torch::Tensor output = glu_module->forward(input);
        
        // Try with different dimensions if there's more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::nn::GLU glu_module2(torch::nn::GLUOptions().dim(dim));
            torch::Tensor output2 = glu_module2->forward(input);
        }
        
        // Try functional version if possible
        try {
            torch::Tensor functional_output = torch::glu(input, dim);
        } catch (const std::exception&) {
            // Ignore exceptions from functional version
        }
        
        // Try with negative dimension
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make dimension negative to test edge case
            dim = -std::abs(dim);
            
            try {
                torch::nn::GLU glu_module_neg(torch::nn::GLUOptions().dim(dim));
                torch::Tensor output_neg = glu_module_neg->forward(input);
            } catch (const std::exception&) {
                // Ignore exceptions from negative dimension
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
