#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early return if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (batch, channels, depth, height, width)
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvReLU3d
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t dilation = 0;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset + 6 <= Size) {
            out_channels = Data[offset++] % 8 + 1; // 1-8 output channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            groups = Data[offset++] % 2 + 1;       // 1-2 groups
            
            // Ensure groups divides both in_channels and out_channels
            int64_t in_channels = input.size(1);
            if (in_channels % groups != 0) {
                in_channels = groups;
                input = input.reshape({input.size(0), in_channels, 
                                      input.size(2), input.size(3), input.size(4)});
            }
            
            // Adjust out_channels to be divisible by groups
            if (out_channels % groups != 0) {
                out_channels = groups * (out_channels / groups + 1);
            }
        }
        
        // Create Conv3d module and apply ReLU manually
        torch::nn::Conv3dOptions options(input.size(1), out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        // Create the Conv3d module
        torch::nn::Conv3d conv(options);
        
        // Convert input to float if needed
        if (input.dtype() != torch::kFloat && 
            input.dtype() != torch::kDouble && 
            input.dtype() != torch::kHalf) {
            input = input.to(torch::kFloat);
        }
        
        // Apply the operation (Conv3d + ReLU)
        torch::Tensor conv_output = conv(input);
        torch::Tensor output = torch::relu(conv_output);
        
        // Verify output has expected properties
        if (output.dim() != 5) {
            throw std::runtime_error("Output dimension mismatch");
        }
        
        // Test with empty input
        if (offset + 1 <= Size && Data[offset] % 2 == 0) {
            try {
                torch::Tensor empty_input = torch::empty({0, input.size(1), 0, 0, 0}, input.options());
                torch::Tensor empty_conv_output = conv(empty_input);
                torch::Tensor empty_output = torch::relu(empty_conv_output);
            } catch (...) {
                // Expected to potentially throw, just catch and continue
            }
        }
        
        // Test with very small input
        if (offset + 1 <= Size && Data[offset] % 3 == 0) {
            try {
                torch::Tensor small_input = torch::ones({1, input.size(1), 1, 1, 1}, input.options());
                torch::Tensor small_conv_output = conv(small_input);
                torch::Tensor small_output = torch::relu(small_conv_output);
            } catch (...) {
                // Expected to potentially throw, just catch and continue
            }
        }
        
        // Test with negative values to ensure ReLU works
        if (offset + 1 <= Size && Data[offset] % 4 == 0) {
            try {
                torch::Tensor neg_input = -torch::ones({1, input.size(1), 3, 3, 3}, input.options());
                torch::Tensor neg_conv_output = conv(neg_input);
                torch::Tensor neg_output = torch::relu(neg_conv_output);
                
                // Verify all values are >= 0 (ReLU effect)
                if (torch::any(neg_output < 0).item<bool>()) {
                    throw std::runtime_error("ReLU failed: negative values in output");
                }
            } catch (...) {
                // Expected to potentially throw, just catch and continue
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