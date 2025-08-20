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
        
        // Extract parameters for rrelu from the remaining data
        double lower = 0.125;
        double upper = 0.3333333333333333;
        bool training = false;
        
        // If we have more data, use it to set parameters
        if (offset + 8 <= Size) {
            // Extract lower bound (between 0 and 1)
            double raw_lower;
            std::memcpy(&raw_lower, Data + offset, sizeof(double));
            offset += sizeof(double);
            lower = std::abs(raw_lower) / (std::abs(raw_lower) + 1.0); // Normalize to [0,1]
        }
        
        if (offset + 8 <= Size) {
            // Extract upper bound (between lower and 1)
            double raw_upper;
            std::memcpy(&raw_upper, Data + offset, sizeof(double));
            offset += sizeof(double);
            upper = lower + (std::abs(raw_upper) / (std::abs(raw_upper) + 1.0)) * (1.0 - lower); // Normalize to [lower,1]
        }
        
        if (offset < Size) {
            // Use one byte to determine training mode
            training = (Data[offset++] % 2 == 0);
        }
        
        // Apply rrelu operation
        torch::Tensor output = torch::rrelu(input, lower, upper, training);
        
        // Try different variants of the API
        if (offset < Size) {
            uint8_t api_variant = Data[offset++] % 3;
            
            switch (api_variant) {
                case 0:
                    // In-place version
                    torch::rrelu_(input, lower, upper, training);
                    break;
                case 1:
                    // Functional interface with default parameters
                    output = torch::rrelu(input);
                    break;
                case 2:
                    // Using torch::nn::functional
                    output = torch::nn::functional::rrelu(input, torch::nn::functional::RReLUFuncOptions().lower(lower).upper(upper).training(training));
                    break;
            }
        }
        
        // Try to access elements to ensure computation is performed
        if (!output.sizes().empty() && output.numel() > 0) {
            auto accessor = output.accessor<float, 1>();
            volatile float first_element = accessor[0];
            (void)first_element;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}