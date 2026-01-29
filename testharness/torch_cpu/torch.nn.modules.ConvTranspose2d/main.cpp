#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some data to proceed
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for ConvTranspose2d from data first
        uint8_t in_channels = (Data[offset++] % 8) + 1;      // 1-8
        uint8_t out_channels = (Data[offset++] % 8) + 1;     // 1-8
        uint8_t kernel_size = (Data[offset++] % 5) + 1;      // 1-5
        uint8_t stride = (Data[offset++] % 3) + 1;           // 1-3
        uint8_t padding = Data[offset++] % 3;                // 0-2
        uint8_t dilation = (Data[offset++] % 2) + 1;         // 1-2
        uint8_t groups_selector = Data[offset++];
        bool bias = Data[offset++] & 1;
        
        // Calculate groups - must divide both in_channels and out_channels
        uint8_t groups = 1;
        if (groups_selector % 4 == 0 && in_channels % 2 == 0 && out_channels % 2 == 0) {
            groups = 2;
        }
        if (groups_selector % 8 == 0 && in_channels % 4 == 0 && out_channels % 4 == 0) {
            groups = 4;
        }
        
        // output_padding must be smaller than either stride or dilation
        uint8_t max_output_padding = std::min(stride, dilation) - 1;
        uint8_t output_padding = (max_output_padding > 0) ? (Data[offset++] % (max_output_padding + 1)) : 0;
        
        // Spatial dimensions from fuzzer data
        uint8_t height = (Data[offset++] % 16) + 1;  // 1-16
        uint8_t width = (Data[offset++] % 16) + 1;   // 1-16
        uint8_t batch = (Data[offset++] % 4) + 1;    // 1-4
        
        // Ensure input spatial size is valid for the convolution
        // For ConvTranspose2d, input can be small, but let's ensure at least 1x1
        int64_t h = std::max((int64_t)height, (int64_t)1);
        int64_t w = std::max((int64_t)width, (int64_t)1);
        
        // Create input tensor with proper shape (N, C_in, H, W)
        torch::Tensor input = torch::randn({batch, in_channels, h, w});
        
        // Use remaining fuzzer data to perturb the tensor values if available
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset] | (Data[offset+1] << 8)) / 65535.0f * 10.0f;
            offset += 2;
            float shift = static_cast<float>(Data[offset] | (Data[offset+1] << 8)) / 65535.0f * 5.0f - 2.5f;
            offset += 2;
            input = input * scale + shift;
        }
        
        // Create ConvTranspose2d module
        torch::nn::ConvTranspose2dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose2d(options);
        
        // Apply the operation
        torch::Tensor output;
        try {
            output = conv_transpose->forward(input);
        } catch (const c10::Error&) {
            // Shape/dimension errors are expected for some parameter combinations
            return 0;
        }
        
        // Ensure the output is used to prevent optimization
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with output_size parameter variant
        if (offset + 2 <= Size && output.dim() == 4) {
            int64_t target_h = output.size(2) + (Data[offset++] % 3);
            int64_t target_w = output.size(3) + (Data[offset++] % 3);
            
            try {
                torch::Tensor output2 = conv_transpose->forward(input, 
                    c10::IntArrayRef({target_h, target_w}));
                volatile float sum2 = output2.sum().item<float>();
                (void)sum2;
            } catch (const c10::Error&) {
                // Invalid output_size for these parameters - expected
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