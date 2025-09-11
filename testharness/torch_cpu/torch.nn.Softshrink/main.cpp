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
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse lambda parameter for Softshrink
        float lambda = 0.5f;  // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&lambda, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure lambda is not NaN or Inf
            if (std::isnan(lambda) || std::isinf(lambda)) {
                lambda = 0.5f;
            }
        }
        
        // Create Softshrink module
        torch::nn::Softshrink softshrink_module(lambda);
        
        // Apply Softshrink operation
        torch::Tensor output = softshrink_module->forward(input);
        
        // Alternative way to apply Softshrink using functional API
        torch::Tensor output2 = torch::nn::functional::softshrink(input, torch::nn::functional::SoftshrinkFuncOptions(lambda));
        
        // Try with different lambda values
        if (offset + sizeof(float) <= Size) {
            float lambda2;
            std::memcpy(&lambda2, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure lambda2 is not NaN or Inf
            if (!std::isnan(lambda2) && !std::isinf(lambda2)) {
                torch::nn::Softshrink softshrink_module2(lambda2);
                torch::Tensor output3 = softshrink_module2->forward(input);
                
                torch::Tensor output4 = torch::nn::functional::softshrink(input, torch::nn::functional::SoftshrinkFuncOptions(lambda2));
            }
        }
        
        // Try with zero lambda
        torch::nn::Softshrink softshrink_zero(0.0);
        torch::Tensor output_zero = softshrink_zero->forward(input);
        
        // Try with negative lambda
        torch::nn::Softshrink softshrink_neg(-lambda);
        torch::Tensor output_neg = softshrink_neg->forward(input);
        
        // Try with very small lambda
        torch::nn::Softshrink softshrink_small(1e-10);
        torch::Tensor output_small = softshrink_small->forward(input);
        
        // Try with very large lambda
        torch::nn::Softshrink softshrink_large(1e10);
        torch::Tensor output_large = softshrink_large->forward(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
