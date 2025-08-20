#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create regular modules
        torch::nn::Linear linear(torch::nn::LinearOptions(10, 10));
        torch::nn::Conv2d conv2d(torch::nn::Conv2dOptions(3, 16, 3));
        
        // Try to apply modules
        if (input.dim() >= 2) {
            try {
                // For Conv2d, we need at least 4D tensor (batch, channels, height, width)
                if (input.dim() >= 4 && input.size(1) == 3) {
                    auto output_conv = conv2d(input);
                }
            } catch (...) {
                // Ignore exceptions from Conv2d
            }
        }
        
        // Try Linear module
        try {
            // Reshape input if needed to match linear layer requirements
            torch::Tensor reshaped_input;
            if (input.dim() == 0) {
                // Scalar tensor - expand to 2D
                reshaped_input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                // 1D tensor - expand to 2D
                reshaped_input = input.unsqueeze(0);
            } else {
                // Use as is, but ensure last dimension matches input features
                reshaped_input = input;
            }
            
            // Ensure the last dimension matches the linear layer's input features
            if (reshaped_input.size(-1) != linear->weight.size(1)) {
                // Resize the last dimension
                std::vector<int64_t> new_shape = reshaped_input.sizes().vec();
                new_shape.back() = linear->weight.size(1);
                reshaped_input = torch::zeros(new_shape, reshaped_input.options());
            }
            
            auto output_linear = linear(reshaped_input);
        } catch (...) {
            // Ignore exceptions from Linear
        }
        
        // Test training functionality
        try {
            linear->train();
            
            // Create a simple input that matches the linear layer's input size
            auto simple_input = torch::ones({1, linear->weight.size(1)});
            auto output = linear(simple_input);
            
            // Test eval mode
            linear->eval();
        } catch (...) {
            // Ignore exceptions from training
        }
        
        // Test BatchNorm
        try {
            torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(16));
            
            // Create a valid input for BatchNorm2d
            auto bn_input = torch::ones({1, 16, 10, 10});
            auto bn_output = bn(bn_input);
        } catch (...) {
            // Ignore exceptions from BatchNorm
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}