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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4 bytes left for parameters
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract parameters for CrossMapLRN2d
        int64_t size_param = Data[offset++] % 10 + 1; // Size parameter between 1 and 10
        
        // Extract alpha, beta, and k parameters
        float alpha = 0.0001f;
        float beta = 0.75f;
        float k = 1.0f;
        
        if (offset + 3 <= Size) {
            // Use remaining bytes to influence parameters
            alpha = static_cast<float>(Data[offset++]) / 1000.0f; // Small alpha value
            beta = static_cast<float>(Data[offset++]) / 100.0f;   // Reasonable beta value
            k = static_cast<float>(Data[offset++]) / 10.0f;       // Reasonable k value
        }
        
        // Ensure input has at least 3 dimensions (batch, channels, height, width)
        // If not, reshape it to a valid shape for CrossMapLRN2d
        if (input.dim() < 3) {
            // Reshape to a minimal valid shape: [1, 3, 3, 3]
            input = input.reshape({1, 3, 3, 3});
        }
        
        // Create CrossMapLRN2d module using options
        torch::nn::CrossMapLRN2dOptions options(size_param);
        options.alpha(alpha).beta(beta).k(k);
        torch::nn::CrossMapLRN2d lrn(options);
        
        // Apply the operation
        torch::Tensor output = lrn->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<float>() == -1.0f) {
            return 1; // This branch is unlikely to be taken
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
