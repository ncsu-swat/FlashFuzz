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
        
        // Parse lambda value from the remaining data
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
        
        // Create Hardshrink module with the parsed lambda
        torch::nn::Hardshrink hardshrink(lambda);
        
        // Apply Hardshrink to the input tensor
        torch::Tensor output = hardshrink(input);
        
        // Alternative way to apply Hardshrink using functional API
        torch::Tensor output2 = torch::nn::functional::hardshrink(input, torch::nn::functional::HardshrinkFuncOptions().lambda(lambda));
        
        // Try with different lambda values to test edge cases
        if (offset + 1 <= Size) {
            uint8_t lambda_selector = Data[offset++];
            
            // Test with zero lambda
            if (lambda_selector % 5 == 0) {
                torch::nn::Hardshrink zero_hardshrink(0.0);
                torch::Tensor zero_output = zero_hardshrink(input);
            }
            
            // Test with very small lambda
            if (lambda_selector % 5 == 1) {
                torch::nn::Hardshrink small_hardshrink(1e-10);
                torch::Tensor small_output = small_hardshrink(input);
            }
            
            // Test with very large lambda
            if (lambda_selector % 5 == 2) {
                torch::nn::Hardshrink large_hardshrink(1e10);
                torch::Tensor large_output = large_hardshrink(input);
            }
            
            // Test with negative lambda (should behave the same as positive)
            if (lambda_selector % 5 == 3) {
                torch::nn::Hardshrink neg_hardshrink(-lambda);
                torch::Tensor neg_output = neg_hardshrink(input);
            }
            
            // Test with NaN lambda (should default to 0.5 or throw)
            if (lambda_selector % 5 == 4) {
                try {
                    double nan_lambda = std::numeric_limits<double>::quiet_NaN();
                    torch::nn::Hardshrink nan_hardshrink(nan_lambda);
                    torch::Tensor nan_output = nan_hardshrink(input);
                } catch (...) {
                    // Expected behavior might be to throw
                }
            }
        }
        
        // Test with cloned input for additional testing
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                torch::Tensor input_clone = input.clone();
                torch::Tensor clone_output = torch::nn::functional::hardshrink(input_clone, torch::nn::functional::HardshrinkFuncOptions().lambda(lambda));
            } catch (...) {
                // Might throw for certain inputs
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