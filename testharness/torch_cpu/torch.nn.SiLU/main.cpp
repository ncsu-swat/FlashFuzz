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
        
        // Create SiLU module
        torch::nn::SiLU silu_module;
        
        // Apply SiLU operation
        torch::Tensor output = silu_module->forward(input);
        
        // Alternative way to apply SiLU using functional API
        torch::Tensor output_functional = torch::nn::functional::silu(input);
        
        // Try with different input types
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply SiLU to the second tensor
            torch::Tensor output2 = silu_module->forward(input2);
            
            // Try with inplace operation if possible
            if (input2.is_floating_point()) {
                torch::Tensor input2_clone = input2.clone();
                torch::nn::functional::silu(input2_clone, torch::nn::functional::SiLUFuncOptions().inplace(true));
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
