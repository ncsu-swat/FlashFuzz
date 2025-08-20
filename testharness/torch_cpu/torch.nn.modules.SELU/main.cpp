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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create SELU module with default parameters
        torch::nn::SELU selu_default;
        
        // Create SELU module with custom parameters if we have more data
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x01; // Use 1 bit to determine inplace
        }
        torch::nn::SELU selu_custom(torch::nn::SELUOptions().inplace(inplace));
        
        // Apply SELU operations
        torch::Tensor output_default = selu_default->forward(input);
        torch::Tensor output_custom = selu_custom->forward(input);
        
        // Try with functional API as well
        torch::Tensor output_functional = torch::selu(input);
        
        // Try with inplace version if we're using inplace mode
        if (inplace) {
            torch::Tensor input_copy = input.clone();
            torch::selu_(input_copy);
        }
        
        // Try with edge cases if we have enough data
        if (offset < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_input;
            
            uint8_t extreme_type = Data[offset++] % 4;
            if (extreme_type == 0) {
                // Very large positive values
                extreme_input = torch::ones_like(input) * 1e10;
            } else if (extreme_type == 1) {
                // Very large negative values
                extreme_input = torch::ones_like(input) * -1e10;
            } else if (extreme_type == 2) {
                // NaN values
                extreme_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
            } else {
                // Inf values
                extreme_input = torch::full_like(input, std::numeric_limits<float>::infinity());
            }
            
            // Apply SELU to extreme values
            torch::Tensor extreme_output = selu_default->forward(extreme_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}