#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::min
#include <cstring>        // For memcpy

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
        } else {
            // Default values if not enough data
            in_channels = 3;
            out_channels = 8;
            kernel_size = 3;
            stride = 1;
            padding = 1;
            dilation = 1;
            groups = 1;
        }
        
        // Ensure input tensor has correct shape for 3D convolution (N, C, D, H, W)
        if (input.dim() != 5 || input.size(1) != in_channels) {
            // Create a new tensor with the right shape
            int64_t depth = 8, height = 8, width = 8;
            if (offset + 3 <= Size) {
                depth = (Data[offset++] % 8) + 4;   // 4-11
                height = (Data[offset++] % 8) + 4;  // 4-11
                width = (Data[offset++] % 8) + 4;   // 4-11
            }
            input = torch::randn({1, in_channels, depth, height, width});
        }
        
        // Ensure input has float dtype for QAT
        if (input.dtype() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Create scale and zero_point for quantization
        float scale = 1.0f;
        int64_t zero_point = 0;
        if (offset + 8 <= Size) {
            // Extract scale from data (ensure it's positive and reasonable)
            float extracted_scale;
            std::memcpy(&extracted_scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::abs(extracted_scale);
            if (scale < 1e-6f || !std::isfinite(scale)) scale = 0.1f;
            if (scale > 100.0f) scale = 100.0f;
            
            // Extract zero_point from data
            int32_t zp_raw;
            std::memcpy(&zp_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            zero_point = std::max(-128, std::min(127, zp_raw));  // Clamp to int8 range
        }
        
        // Create Conv3d and ReLU modules
        // Note: torch.nn.intrinsic.qat.modules.ConvReLU3d is Python-only
        // This harness tests the underlying Conv3d + ReLU + fake_quantize operations
        torch::nn::Conv3dOptions conv_options(in_channels, out_channels, kernel_size);
        conv_options.stride(stride)
                   .padding(padding)
                   .dilation(dilation)
                   .groups(groups)
                   .bias(bias);
        
        auto conv3d = torch::nn::Conv3d(conv_options);
        
        // Apply Conv3d
        torch::Tensor conv_output;
        try {
            conv_output = conv3d->forward(input);
        } catch (...) {
            // Shape mismatch or other issues with conv parameters
            return 0;
        }
        
        // Apply ReLU (inplace variant for efficiency, matching fused behavior)
        torch::Tensor relu_output = torch::relu(conv_output);
        
        // Test quantization-aware training simulation with fake_quantize
        // This simulates what ConvReLU3d does in QAT mode
        torch::Tensor fake_quantized = torch::fake_quantize_per_tensor_affine(
            relu_output, scale, zero_point, -128, 127);
        
        // Additional coverage: test with different quant parameters
        if (offset + 4 <= Size) {
            float scale2;
            std::memcpy(&scale2, Data + offset, sizeof(float));
            scale2 = std::abs(scale2);
            if (scale2 >= 1e-6f && scale2 <= 100.0f && std::isfinite(scale2)) {
                torch::Tensor fake_quantized2 = torch::fake_quantize_per_tensor_affine(
                    relu_output, scale2, 0, -128, 127);
                (void)fake_quantized2;
            }
        }
        
        // Test per-channel fake quantization (also used in QAT)
        if (out_channels > 0) {
            try {
                torch::Tensor scales = torch::ones({out_channels}) * scale;
                torch::Tensor zero_points = torch::zeros({out_channels}, torch::kLong);
                torch::Tensor fake_quantized_channel = torch::fake_quantize_per_channel_affine(
                    relu_output, scales, zero_points, 1, -128, 127);
                (void)fake_quantized_channel;
            } catch (...) {
                // Per-channel quantization may fail with certain shapes
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}