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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create SiLU module
        torch::nn::SiLU silu_module;
        
        // Apply SiLU operation
        torch::Tensor output = silu_module->forward(input);
        
        // Alternative way to apply SiLU using functional API
        torch::Tensor output_functional = torch::nn::functional::silu(input);
        
        // Try inplace version if available
        if (input.is_floating_point() && input.requires_grad() == false) {
            torch::Tensor input_copy = input.clone();
            torch::nn::functional::silu(input_copy, torch::nn::functional::SiLUFuncOptions().inplace(true));
        }
        
        // Test with different configurations
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            
            // Create another SiLU module
            torch::nn::SiLU silu_with_options;
            
            // Apply the module
            torch::Tensor result;
            if (inplace && input.is_floating_point() && !input.requires_grad()) {
                result = silu_with_options->forward(input.clone());
            } else {
                result = silu_with_options->forward(input);
            }
        }
        
        // Test with edge cases if we have enough data
        if (offset + 1 < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_input;
            
            uint8_t extreme_type = Data[offset++] % 3;
            if (extreme_type == 0 && input.is_floating_point()) {
                // Very large values
                extreme_input = torch::full_like(input, 1e38);
            } else if (extreme_type == 1 && input.is_floating_point()) {
                // Very small values
                extreme_input = torch::full_like(input, -1e38);
            } else {
                // NaN and Inf values for floating point tensors
                if (input.is_floating_point()) {
                    extreme_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                    torch::Tensor inf_input = torch::full_like(input, std::numeric_limits<float>::infinity());
                    torch::nn::SiLU silu_temp;
                    silu_temp->forward(inf_input);
                } else {
                    // For non-floating point, use the original input
                    extreme_input = input;
                }
            }
            
            // Apply SiLU to extreme values
            torch::nn::SiLU silu_extreme;
            silu_extreme->forward(extreme_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
