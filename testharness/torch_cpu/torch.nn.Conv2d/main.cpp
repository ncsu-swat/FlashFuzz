#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for Conv2d first
        uint8_t in_channels = Data[offset++] % 16 + 1;
        uint8_t out_channels = Data[offset++] % 16 + 1;
        uint8_t kernel_size = Data[offset++] % 5 + 1;
        uint8_t stride = Data[offset++] % 3 + 1;
        uint8_t padding = Data[offset++] % 3;
        uint8_t dilation = Data[offset++] % 2 + 1;
        bool use_bias = Data[offset++] % 2 == 0;
        
        // Determine spatial dimensions from fuzz data
        uint8_t height = (offset < Size) ? (Data[offset++] % 16 + kernel_size * dilation) : 8;
        uint8_t width = (offset < Size) ? (Data[offset++] % 16 + kernel_size * dilation) : 8;
        uint8_t batch_size = (offset < Size) ? (Data[offset++] % 4 + 1) : 1;
        
        // Create properly shaped input tensor (N, C, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});
        
        // Seed randomness from fuzz data for tensor values
        if (offset < Size) {
            float scale = (Data[offset++] % 100) / 10.0f + 0.1f;
            input = input * scale;
        }
        
        // Create and apply Conv2d module
        {
            torch::nn::Conv2d conv(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .bias(use_bias)
            );
            
            try {
                torch::Tensor output = conv->forward(input);
            } catch (...) {
                // Expected failure due to invalid parameters
            }
        }
        
        // Try groups parameter
        if (offset < Size) {
            uint8_t groups = Data[offset++] % std::min(in_channels, out_channels) + 1;
            // Find valid groups value
            while (groups > 1 && (in_channels % groups != 0 || out_channels % groups != 0)) {
                groups--;
            }
            
            try {
                torch::nn::Conv2d conv_grouped(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .groups(groups)
                        .bias(use_bias)
                );
                torch::Tensor output = conv_grouped->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Try padding_mode
        if (offset < Size) {
            uint8_t padding_mode_selector = Data[offset++] % 4;
            
            try {
                torch::nn::Conv2dOptions opts(in_channels, out_channels, kernel_size);
                opts.stride(stride).padding(padding).dilation(dilation).bias(use_bias);
                
                switch (padding_mode_selector) {
                    case 0: opts.padding_mode(torch::kZeros); break;
                    case 1: opts.padding_mode(torch::kReflect); break;
                    case 2: opts.padding_mode(torch::kReplicate); break;
                    case 3: opts.padding_mode(torch::kCircular); break;
                }
                
                torch::nn::Conv2d conv_pm(opts);
                torch::Tensor output = conv_pm->forward(input);
            } catch (...) {
                // Expected failure (e.g., reflect with large padding)
            }
        }
        
        // Try different kernel sizes for height and width
        if (offset + 1 < Size) {
            uint8_t kernel_h = Data[offset++] % 5 + 1;
            uint8_t kernel_w = Data[offset++] % 5 + 1;
            
            try {
                torch::nn::Conv2d conv_ksize(
                    torch::nn::Conv2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .bias(use_bias)
                );
                torch::Tensor output = conv_ksize->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Try different strides for height and width
        if (offset + 1 < Size) {
            uint8_t stride_h = Data[offset++] % 3 + 1;
            uint8_t stride_w = Data[offset++] % 3 + 1;
            
            try {
                torch::nn::Conv2d conv_stride(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride({stride_h, stride_w})
                        .padding(padding)
                        .dilation(dilation)
                        .bias(use_bias)
                );
                torch::Tensor output = conv_stride->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Try different paddings for height and width
        if (offset + 1 < Size) {
            uint8_t padding_h = Data[offset++] % 4;
            uint8_t padding_w = Data[offset++] % 4;
            
            try {
                torch::nn::Conv2d conv_pad(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding({padding_h, padding_w})
                        .dilation(dilation)
                        .bias(use_bias)
                );
                torch::Tensor output = conv_pad->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Try different dilations for height and width
        if (offset + 1 < Size) {
            uint8_t dilation_h = Data[offset++] % 3 + 1;
            uint8_t dilation_w = Data[offset++] % 3 + 1;
            
            try {
                torch::nn::Conv2d conv_dil(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation({dilation_h, dilation_w})
                        .bias(use_bias)
                );
                torch::Tensor output = conv_dil->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Try double precision
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                torch::Tensor input_double = input.to(torch::kFloat64);
                torch::nn::Conv2d conv_double(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .bias(use_bias)
                );
                conv_double->to(torch::kFloat64);
                torch::Tensor output = conv_double->forward(input_double);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Test with zero-sized batch
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                torch::Tensor empty_input = torch::randn({0, in_channels, height, width});
                torch::nn::Conv2d conv_empty(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .bias(use_bias)
                );
                torch::Tensor output = conv_empty->forward(empty_input);
            } catch (...) {
                // Expected failure
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // Keep the input
}