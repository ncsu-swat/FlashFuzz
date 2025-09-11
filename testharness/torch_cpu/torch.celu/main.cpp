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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract alpha parameter from the remaining data if available
        float alpha = 1.0f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure alpha is not zero or negative (which would cause issues)
            if (alpha <= 0.0f) {
                alpha = 1.0f;
            }
        }
        
        // Apply CELU activation function
        torch::Tensor output = torch::celu(input, alpha);
        
        // Try different variants of the API
        if (Size > offset) {
            uint8_t variant = Data[offset++];
            
            // Test in-place version
            if (variant % 3 == 0 && input.is_floating_point()) {
                torch::Tensor input_copy = input.clone();
                torch::celu_(input_copy, alpha);
            }
            
            // Test functional version with options
            if (variant % 3 == 1) {
                auto options = torch::nn::functional::CELUFuncOptions().alpha(alpha);
                torch::Tensor output2 = torch::nn::functional::celu(input, options);
            }
            
            // Test module version
            if (variant % 3 == 2) {
                torch::nn::CELU celu_module(torch::nn::CELUOptions().alpha(alpha));
                torch::Tensor output3 = celu_module->forward(input);
            }
        }
        
        // Test with extreme values for alpha
        if (Size > offset + sizeof(float)) {
            float extreme_alpha;
            std::memcpy(&extreme_alpha, Data + offset, sizeof(float));
            
            // Don't filter out extreme values - let PyTorch handle them
            torch::Tensor output_extreme = torch::celu(input, extreme_alpha);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
