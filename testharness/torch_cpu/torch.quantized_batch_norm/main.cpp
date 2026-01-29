#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need sufficient data
        if (Size < 16) {
            return 0;
        }
        
        // Extract scale and zero_point first
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale < 1e-6 || !std::isfinite(scale)) scale = 0.1;
            if (scale > 1e6) scale = 1e6;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = std::abs(zero_point) % 256;
        }
        
        // Extract dimensions for batch norm (N, C, H, W) or (N, C, L)
        int64_t batch_size = 1 + (Data[offset % Size] % 4);
        int64_t num_channels = 1 + (Data[(offset + 1) % Size] % 8);
        int64_t height = 1 + (Data[(offset + 2) % Size] % 8);
        int64_t width = 1 + (Data[(offset + 3) % Size] % 8);
        offset += 4;
        
        // Create input tensor - batch norm expects at least 2D with channels
        torch::Tensor input = torch::rand({batch_size, num_channels, height, width});
        
        // Quantize input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // Quantization failed, skip this input
            return 0;
        }
        
        // Create weight, bias, running_mean, and running_var
        // These should be 1D tensors with size matching num_channels
        torch::Tensor weight = torch::ones({num_channels});
        torch::Tensor bias = torch::zeros({num_channels});
        torch::Tensor running_mean = torch::zeros({num_channels});
        torch::Tensor running_var = torch::ones({num_channels});
        
        // Vary weight and bias using fuzzer data
        if (offset + num_channels <= Size) {
            for (int64_t i = 0; i < num_channels && offset < Size; i++, offset++) {
                float w = 0.5f + (Data[offset] % 100) / 100.0f;
                weight[i] = w;
            }
        }
        
        if (offset + num_channels <= Size) {
            for (int64_t i = 0; i < num_channels && offset < Size; i++, offset++) {
                float b = (Data[offset] % 100) / 100.0f - 0.5f;
                bias[i] = b;
            }
        }
        
        // Extract epsilon
        double epsilon = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&epsilon, Data + offset, sizeof(double));
            offset += sizeof(double);
            epsilon = std::abs(epsilon);
            if (epsilon < 1e-10 || !std::isfinite(epsilon)) epsilon = 1e-5;
            if (epsilon > 0.1) epsilon = 0.1;
        }
        
        // Output scale and zero_point
        double output_scale = scale;
        int64_t output_zero_point = zero_point;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&output_scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            output_scale = std::abs(output_scale);
            if (output_scale < 1e-6 || !std::isfinite(output_scale)) output_scale = 0.1;
            if (output_scale > 1e6) output_scale = 1e6;
        }
        
        // Apply quantized_batch_norm
        torch::Tensor output;
        try {
            output = torch::quantized_batch_norm(
                q_input, 
                weight, 
                bias, 
                running_mean, 
                running_var, 
                epsilon, 
                output_scale, 
                output_zero_point
            );
        } catch (...) {
            // Expected failures due to shape/dtype mismatches
            return 0;
        }
        
        // Verify output tensor is valid
        auto output_sizes = output.sizes();
        (void)output_sizes;
        
        // Dequantize to check correctness
        try {
            torch::Tensor dequantized = output.dequantize();
            (void)dequantized.sum().item<float>();
        } catch (...) {
            // Dequantization may fail for some quantized types
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}