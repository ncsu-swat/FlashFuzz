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
        
        // Extract alpha parameter from the remaining data
        double alpha = 1.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure alpha is positive as required by CELU
            alpha = std::abs(alpha);
            
            // Avoid extremely large values that might cause numerical issues
            if (alpha > 1e6) {
                alpha = 1e6;
            }
            if (alpha < 1e-6) {
                alpha = 1e-6;
            }
        }
        
        // Create CELU module
        torch::nn::CELU celu_module(torch::nn::CELUOptions().alpha(alpha));
        
        // Apply CELU operation
        torch::Tensor output = celu_module->forward(input);
        
        // Alternative: use the functional version
        torch::Tensor output_functional = torch::nn::functional::celu(input, torch::nn::functional::CELUFuncOptions().alpha(alpha));
        
        // Try inplace version if available
        if (input.is_floating_point() && input.is_contiguous()) {
            torch::Tensor input_copy = input.clone();
            input_copy.celu_(alpha);
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
