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
        
        // Need at least a few bytes to create a tensor and lambda value
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract lambda value from the input data
        double lambda = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            float lambda_raw;
            std::memcpy(&lambda_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure lambda is positive (as typically expected)
            lambda = std::abs(lambda_raw);
            
            // Limit to reasonable range to avoid extreme values
            lambda = std::fmod(lambda, 1000.0);
        }
        
        // Create Softshrink module
        torch::nn::Softshrink softshrink_module(lambda);
        
        // Apply Softshrink operation
        torch::Tensor output = softshrink_module->forward(input);
        
        // Try with different lambda values to test edge cases
        if (offset + sizeof(float) <= Size) {
            float lambda2_raw;
            std::memcpy(&lambda2_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Try with zero lambda
            torch::nn::Softshrink softshrink_zero(0.0);
            torch::Tensor output_zero = softshrink_zero->forward(input);
            
            // Try with negative lambda (should behave like positive lambda)
            double neg_lambda = -std::abs(lambda2_raw);
            torch::nn::Softshrink softshrink_neg(neg_lambda);
            torch::Tensor output_neg = softshrink_neg->forward(input);
            
            // Try with very small lambda
            torch::nn::Softshrink softshrink_small(1e-10);
            torch::Tensor output_small = softshrink_small->forward(input);
            
            // Try with very large lambda
            torch::nn::Softshrink softshrink_large(1e10);
            torch::Tensor output_large = softshrink_large->forward(input);
        }
        
        // Test functional version as well
        torch::Tensor output_functional = torch::nn::functional::softshrink(input, torch::nn::functional::SoftshrinkFuncOptions(lambda));
        
        // Test with empty tensor
        if (input.numel() > 0) {
            torch::Tensor empty_tensor = torch::empty({0}, input.options());
            torch::Tensor output_empty = softshrink_module->forward(empty_tensor);
        }
        
        // Test with different dtypes if input is floating point
        if (input.is_floating_point()) {
            // Test with half precision
            torch::Tensor input_half = input.to(torch::kHalf);
            torch::Tensor output_half = softshrink_module->forward(input_half);
            
            // Test with double precision
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::Tensor output_double = softshrink_module->forward(input_double);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
