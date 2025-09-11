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
        
        // Create Softsign module
        torch::nn::Softsign softsign_module;
        
        // Apply Softsign operation
        torch::Tensor output = softsign_module->forward(input);
        
        // Alternative way to apply Softsign using functional API
        torch::Tensor output_functional = torch::nn::functional::softsign(input);
        
        // Test with different tensor properties
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply Softsign to the second tensor
            torch::Tensor output2 = softsign_module->forward(input2);
        }
        
        // Test with edge cases if we have enough data
        if (offset + 1 < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Multiply by a large value to create potential overflow
            extreme_input = extreme_input * 1e10;
            
            // Apply Softsign to extreme values
            torch::Tensor extreme_output = softsign_module->forward(extreme_input);
        }
        
        // Test with zero tensor
        if (input.numel() > 0) {
            torch::Tensor zero_input = torch::zeros_like(input);
            torch::Tensor zero_output = softsign_module->forward(zero_input);
        }
        
        // Test with very small values
        if (input.numel() > 0) {
            torch::Tensor small_input = input * 1e-10;
            torch::Tensor small_output = softsign_module->forward(small_input);
        }
        
        // Test with NaN and Inf values if tensor is floating point
        if (input.is_floating_point() && input.numel() > 0) {
            torch::Tensor special_input = input.clone();
            
            // Set some elements to NaN and Inf
            if (special_input.numel() > 2) {
                special_input.index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                special_input.index_put_({special_input.numel()-1}, std::numeric_limits<float>::infinity());
                
                // Apply Softsign to special values
                torch::Tensor special_output = softsign_module->forward(special_input);
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
