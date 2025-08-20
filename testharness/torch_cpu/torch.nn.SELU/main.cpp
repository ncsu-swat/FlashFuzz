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
        
        // Create SELU module
        torch::nn::SELU selu_module;
        
        // Apply SELU operation
        torch::Tensor output = selu_module->forward(input);
        
        // Alternative way to apply SELU using functional API
        torch::Tensor output_functional = torch::selu(input);
        
        // Try with different alpha and scale values using functional API
        if (offset + 16 <= Size) {
            double alpha, scale;
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure alpha and scale are valid (non-NaN, non-Inf)
            if (!std::isfinite(alpha) || !std::isfinite(scale)) {
                alpha = 1.6732632423543772848170429916717;
                scale = 1.0507009873554804934193349852946;
            }
            
            // Use functional API with custom alpha and scale
            torch::Tensor custom_output = torch::nn::functional::selu(input, 
                torch::nn::functional::SELUFuncOptions().alpha(alpha).scale(scale));
        }
        
        // Test inplace version
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            torch::selu_(input_copy);
        }
        
        // Test with different input shapes and types
        if (offset + 2 <= Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor another_output = selu_module->forward(another_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}