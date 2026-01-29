#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least a few bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters from fuzzer data
        uint8_t shape_selector = Data[offset++];
        uint8_t padding_selector = Data[offset++];
        
        // Extract dimension sizes (constrained to reasonable values)
        int64_t batch_size = 1 + (Data[offset++] % 8);      // 1-8
        int64_t channels = 1 + (Data[offset++] % 16);        // 1-16
        int64_t width = 4 + (Data[offset++] % 60);           // 4-63 (need minimum for reflection)
        
        // Extract padding values - constrain to valid range for reflection padding
        // Padding must be less than input width
        int64_t max_padding = std::max((int64_t)1, width - 1);
        int64_t padding_left = Data[offset++] % max_padding;
        int64_t padding_right = Data[offset++] % max_padding;
        
        // Create input tensor with appropriate shape for ReflectionPad1d
        // ReflectionPad1d expects 2D (C, W) or 3D (N, C, W) input
        torch::Tensor input;
        if (shape_selector & 0x01) {
            // 3D input: (N, C, W)
            input = torch::randn({batch_size, channels, width});
        } else {
            // 2D input: (C, W)
            input = torch::randn({channels, width});
        }
        
        // Test different padding configurations
        try {
            if (padding_selector & 0x01) {
                // Symmetric padding (single int)
                auto options = torch::nn::ReflectionPad1dOptions(padding_left);
                torch::nn::ReflectionPad1d reflection_pad(options);
                torch::Tensor output = reflection_pad->forward(input);
                output.sum().item<float>();
            } else {
                // Asymmetric padding (pair of ints)
                auto options = torch::nn::ReflectionPad1dOptions({padding_left, padding_right});
                torch::nn::ReflectionPad1d reflection_pad(options);
                torch::Tensor output = reflection_pad->forward(input);
                output.sum().item<float>();
            }
        } catch (const c10::Error &e) {
            // Expected errors from invalid padding configurations - silently ignore
        }
        
        // Also test the functional interface
        try {
            torch::Tensor output_functional = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions({padding_left, padding_right})
                    .mode(torch::kReflect)
            );
            output_functional.sum().item<float>();
        } catch (const c10::Error &e) {
            // Expected errors from invalid configurations - silently ignore
        }
        
        // Test with different dtypes if we have more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::Tensor typed_input;
            
            try {
                if (dtype_selector % 3 == 0) {
                    typed_input = torch::randn({channels, width}, torch::kFloat32);
                } else if (dtype_selector % 3 == 1) {
                    typed_input = torch::randn({channels, width}, torch::kFloat64);
                } else {
                    typed_input = torch::randn({channels, width}, torch::kFloat16);
                }
                
                auto options = torch::nn::ReflectionPad1dOptions({padding_left, padding_right});
                torch::nn::ReflectionPad1d reflection_pad(options);
                torch::Tensor output = reflection_pad->forward(typed_input);
                output.sum().item<float>();
            } catch (const c10::Error &e) {
                // Some dtypes may not be supported - silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}