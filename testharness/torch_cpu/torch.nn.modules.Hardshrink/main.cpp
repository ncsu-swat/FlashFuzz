#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract lambda parameter for Hardshrink if we have more data
        double lambda = 0.5; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&lambda, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure lambda is a reasonable value
            lambda = std::abs(lambda);
            if (std::isnan(lambda) || std::isinf(lambda)) {
                lambda = 0.5;
            }
        }
        
        // Create Hardshrink module with the lambda parameter
        torch::nn::Hardshrink hardshrink_module(lambda);
        
        // Apply Hardshrink operation
        torch::Tensor output = hardshrink_module->forward(input);
        
        // Try functional version as well
        torch::Tensor output_functional = torch::hardshrink(input, lambda);
        
        // Try inplace version if available
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_clone = input.clone();
            input_clone = torch::hardshrink(input_clone, lambda);
        }
        
        // Try with different lambda values if we have more data
        if (offset + sizeof(double) <= Size) {
            double another_lambda;
            std::memcpy(&another_lambda, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure lambda is a reasonable value
            another_lambda = std::abs(another_lambda);
            if (!std::isnan(another_lambda) && !std::isinf(another_lambda)) {
                torch::nn::Hardshrink another_hardshrink(another_lambda);
                torch::Tensor another_output = another_hardshrink->forward(input);
            }
        }
        
        // Try with zero lambda
        torch::nn::Hardshrink zero_hardshrink(0.0);
        torch::Tensor zero_output = zero_hardshrink->forward(input);
        
        // Try with negative lambda (should be handled by abs() in the implementation)
        if (offset < Size) {
            double neg_lambda = -1.0 * std::abs(static_cast<double>(Data[offset]));
            torch::nn::Hardshrink neg_hardshrink(neg_lambda);
            torch::Tensor neg_output = neg_hardshrink->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}