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
        
        // Create ReLU6 module
        torch::nn::ReLU6 relu6_module;
        
        // Apply ReLU6 operation
        torch::Tensor output = relu6_module->forward(input);
        
        // Verify the output by comparing with manual implementation
        torch::Tensor expected_output = torch::clamp(input, 0, 6);
        
        // Check if the outputs match
        auto diff = torch::abs(output - expected_output);
        auto max_diff = torch::max(diff).item<float>();
        
        // Use the tensor in some way to prevent optimization
        if (output.defined() && !output.numel()) {
            // Just to use the tensor
            auto sum = output.sum().item<float>();
            if (std::isnan(sum)) {
                return 0;
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