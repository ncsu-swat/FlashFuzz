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
        
        // Parse lambda value from remaining data
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
        
        // Apply hardshrink operation
        torch::Tensor output = torch::hardshrink(input, lambda);
        
        // Try with named parameters
        torch::Tensor output2 = torch::hardshrink(input, torch::Scalar(lambda));
        
        // Try functional variant
        torch::Tensor output3 = torch::nn::functional::hardshrink(input, torch::nn::functional::HardshrinkFuncOptions().lambda(lambda));
        
        // Create a Hardshrink module and apply it
        torch::nn::Hardshrink hardshrink_module(torch::nn::HardshrinkOptions().lambda(lambda));
        torch::Tensor output4 = hardshrink_module->forward(input);
        
        // Try with edge case lambda values
        if (offset + 1 <= Size) {
            uint8_t lambda_selector = Data[offset++];
            
            // Try with different lambda values based on the selector
            double special_lambda = 0.0;
            switch (lambda_selector % 5) {
                case 0: special_lambda = 0.0; break;
                case 1: special_lambda = std::numeric_limits<double>::min(); break;
                case 2: special_lambda = std::numeric_limits<double>::epsilon(); break;
                case 3: special_lambda = 1.0; break;
                case 4: special_lambda = 100.0; break;
            }
            
            torch::Tensor output5 = torch::hardshrink(input, special_lambda);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
