#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::min
#include <cstring>        // For memcpy

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0;  // Need minimum data for meaningful fuzzing
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for ConvReLU3d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        int kernel_size = 0, stride = 0, padding = 0, dilation = 0, groups = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 16 + 1;  // 1-16 channels
            out_channels = Data[offset++] % 16 + 1; // 1-16 channels
            kernel_size = Data[offset++] % 5 + 1;   // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;        // 1-3 stride
            padding = Data[offset++] % 3;           // 0-2 padding
            dilation = Data[offset++] % 2 + 1;      // 1-2 dilation
            groups = (Data[offset++] % 4) + 1;      // 1-4 groups
            
            // Ensure groups compatibility
            if (in_channels % groups != 0 || out_channels % groups != 0) {
                groups = 1;  // Default to 1 if incompatible
            }
            
            // Determine if bias should be used
            if (offset < Size) {
                bias = Data[offset++] % 2 == 0;
            }
        }
        
        // Ensure input tensor has correct shape for 3D convolution (N, C, D, H, W)
        if (input.dim() != 5) {
            // Reshape to 5D tensor if needed
            std::vector<int64_t> new_shape;
            if (input.dim() < 5) {
                new_shape = {1, in_channels, 8, 8, 8};  // Default shape
                if (input.numel() > 0) {
                    // Try to preserve some dimensions
                    for (int i = 0; i < std::min(static_cast<int>(input.dim()), 5); i++) {
                        if (i == 1) {
                            new_shape[i] = in_channels;  // Force channel dimension
                        } else if (i > 0) {
                            new_shape[i] = std::max<int64_t>(1, std::min<int64_t>(input.size(i-1), 16));
                        }
                    }
                }
            } else {
                // Too many dimensions, slice to 5D
                new_shape = {input.size(0), in_channels, 
                             std::max<int64_t>(1, input.size(2)), 
                             std::max<int64_t>(1, input.size(3)), 
                             std::max<int64_t>(1, input.size(4))};
            }
            
            // Create a new tensor with the right shape
            input = torch::ones(new_shape, input.options());
        } else if (input.size(1) != in_channels) {
            // Fix channel dimension if needed
            auto shape = input.sizes().vec();
            shape[1] = in_channels;
            input = torch::ones(shape, input.options());
        }
        
        // Ensure input has float dtype for QAT
        if (input.dtype() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Create scale and zero_point for quantization
        float scale = 1.0f;
        int zero_point = 0;
        if (offset + 8 <= Size) {
            // Extract scale from data (ensure it's positive)
            float extracted_scale;
            std::memcpy(&extracted_scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::abs(extracted_scale);
            if (scale < 1e-6f) scale = 1e-6f;  // Avoid too small scale
            
            // Extract zero_point from data
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
            zero_point = std::max(-128, std::min(127, zero_point));  // Clamp to int8 range
        }
        
        // Create Conv3d and ReLU modules separately since intrinsic QAT modules may not be available
        torch::nn::Conv3dOptions conv_options(in_channels, out_channels, kernel_size);
        conv_options.stride(stride)
                   .padding(padding)
                   .dilation(dilation)
                   .groups(groups)
                   .bias(bias);
        
        auto conv3d = torch::nn::Conv3d(conv_options);
        auto relu = torch::nn::ReLU();
        
        // Apply the modules to the input tensor
        torch::Tensor conv_output = conv3d->forward(input);
        torch::Tensor output = relu->forward(conv_output);
        
        // Test quantization-aware training simulation
        torch::Tensor fake_quantized = torch::fake_quantize_per_tensor_affine(
            output, scale, zero_point, -128, 127);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}