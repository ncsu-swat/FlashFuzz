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
        if (Size < 8) {
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
        double alpha = 0.0001;
        double beta = 0.75;
        double k = 1.0;
        
        if (offset + 3 <= Size) {
            // Use remaining bytes to influence parameters
            alpha = static_cast<double>(Data[offset++]) / 1000.0; // Small alpha value
            beta = static_cast<double>(Data[offset++]) / 100.0;   // Beta value
            k = static_cast<double>(Data[offset++]) / 10.0;       // k value
        }
        
        // Create CrossMapLRN2d module using options
        torch::nn::CrossMapLRN2dOptions options(size_param);
        options.alpha(alpha).beta(beta).k(k);
        torch::nn::CrossMapLRN2d lrn(options);
        
        // Apply the operation
        torch::Tensor output = lrn->forward(input);
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
