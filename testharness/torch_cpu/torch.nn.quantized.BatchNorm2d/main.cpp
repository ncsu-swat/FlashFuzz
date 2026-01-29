#include "fuzzer_utils.h" // General fuzzing utilities
#include <torch/torch.h>
#include <iostream>       // For cerr

// Note: torch::nn::quantized::BatchNorm2d is not available in the C++ frontend.
// This harness simulates quantized BatchNorm2d by using regular BatchNorm2d
// with manual quantization/dequantization operations.

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
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract num_features first (before creating tensor)
        int64_t num_features = 1 + (Data[offset++] % 64);
        
        // Extract eps parameter (small positive value for numerical stability)
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps = std::abs(eps_raw);
            if (eps < 1e-10 || std::isnan(eps) || std::isinf(eps)) eps = 1e-5;
            if (eps > 0.1) eps = 0.1;
        }
        
        // Extract momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            momentum = std::abs(momentum_raw);
            if (std::isnan(momentum) || std::isinf(momentum)) momentum = 0.1;
            if (momentum > 1.0) momentum = 1.0;
        }
        
        // Extract batch and spatial dimensions
        int64_t batch_size = 1 + (offset < Size ? Data[offset++] % 4 : 0);
        int64_t height = 1 + (offset < Size ? Data[offset++] % 8 : 1);
        int64_t width = 1 + (offset < Size ? Data[offset++] % 8 : 1);
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .track_running_stats(true));
        bn->eval(); // Set to eval mode to avoid running stats issues
        
        // Create input tensor with proper shape [N, C, H, W]
        torch::Tensor input_tensor = torch::randn({batch_size, num_features, height, width});
        
        // Use remaining data to influence tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining, static_cast<size_t>(input_tensor.numel()));
            auto accessor = input_tensor.accessor<float, 4>();
            size_t idx = 0;
            for (int64_t n = 0; n < batch_size && idx < num_elements; n++) {
                for (int64_t c = 0; c < num_features && idx < num_elements; c++) {
                    for (int64_t h = 0; h < height && idx < num_elements; h++) {
                        for (int64_t w = 0; w < width && idx < num_elements; w++) {
                            accessor[n][c][h][w] = static_cast<float>(Data[offset + idx]) / 128.0f - 1.0f;
                            idx++;
                        }
                    }
                }
            }
        }
        
        // Extract quantization parameters
        float scale = 1.0f / 128.0f;
        int64_t zero_point = 128;
        
        if (offset < Size) {
            uint8_t scale_byte = Data[offset++];
            scale = (scale_byte + 1) / 256.0f; // Scale between ~0.004 and 1.0
        }
        if (offset < Size) {
            zero_point = Data[offset++];
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor,
                scale,
                zero_point,
                torch::kQUInt8
            );
        } catch (...) {
            // If quantization fails, return early
            return 0;
        }
        
        // Dequantize, apply BatchNorm, re-quantize (simulating quantized BatchNorm)
        torch::Tensor dequantized_input = quantized_input.dequantize();
        torch::Tensor bn_output = bn(dequantized_input);
        
        // Re-quantize the output
        torch::Tensor quantized_output;
        try {
            quantized_output = torch::quantize_per_tensor(
                bn_output, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } catch (...) {
            // Output quantization might fail if values are out of range
            return 0;
        }
        
        // Test additional operations on the quantized output
        if (offset < Size) {
            uint8_t op_selector = Data[offset++];
            
            switch (op_selector % 5) {
                case 0:
                    // Test dequantization
                    {
                        torch::Tensor dequantized = quantized_output.dequantize();
                        (void)dequantized.sum();
                    }
                    break;
                    
                case 1:
                    // Test q_scale and q_zero_point
                    {
                        double out_scale = quantized_output.q_scale();
                        int64_t out_zero_point = quantized_output.q_zero_point();
                        (void)out_scale;
                        (void)out_zero_point;
                    }
                    break;
                    
                case 2:
                    // Test int_repr
                    {
                        torch::Tensor int_repr = quantized_output.int_repr();
                        (void)int_repr.sum();
                    }
                    break;
                    
                case 3:
                    // Test with training mode
                    try {
                        bn->train();
                        torch::Tensor train_output = bn(dequantized_input);
                        (void)train_output.sum();
                    } catch (...) {
                        // Training mode might fail with small batches
                    }
                    break;
                    
                case 4:
                    // Test clone and operations
                    {
                        torch::Tensor cloned = quantized_output.clone();
                        torch::Tensor dq = cloned.dequantize();
                        (void)dq.mean();
                    }
                    break;
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