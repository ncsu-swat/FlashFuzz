#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (batch, channels, d, h, w)
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for FractionalMaxPool3d
        double output_ratio_d = 0.5;
        double output_ratio_h = 0.5;
        double output_ratio_w = 0.5;
        
        // Use remaining data to set parameters if available
        if (offset + 3 <= Size) {
            output_ratio_d = static_cast<double>(Data[offset++]) / 255.0 * 0.8 + 0.1; // Range [0.1, 0.9]
            output_ratio_h = static_cast<double>(Data[offset++]) / 255.0 * 0.8 + 0.1; // Range [0.1, 0.9]
            output_ratio_w = static_cast<double>(Data[offset++]) / 255.0 * 0.8 + 0.1; // Range [0.1, 0.9]
        }
        
        // Create the FractionalMaxPool3d module
        torch::nn::FractionalMaxPool3d pool(
            torch::nn::FractionalMaxPool3dOptions(std::vector<double>{output_ratio_d, output_ratio_h, output_ratio_w})
        );
        
        // Apply the operation
        auto output = pool->forward(input);
        
        // Perform some operation on the result to ensure it's used
        auto sum = output.sum();
        if (sum.item<float>() == -1.0f) { // Unlikely condition to prevent optimization
            std::cerr << "Unexpected sum value" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}