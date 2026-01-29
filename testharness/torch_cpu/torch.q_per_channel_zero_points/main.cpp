#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need enough data to create tensors and parameters
        if (Size < 16) {
            return 0;
        }
        
        // Create a float tensor that we'll quantize
        torch::Tensor float_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float and has at least 1 dimension
        if (float_tensor.dim() == 0) {
            float_tensor = float_tensor.unsqueeze(0);
        }
        float_tensor = float_tensor.to(torch::kFloat32).contiguous();
        
        // Determine the quantization axis from fuzzer data
        int64_t axis = 0;
        if (offset < Size) {
            axis = static_cast<int64_t>(Data[offset++]) % float_tensor.dim();
        }
        
        // Create scales and zero_points for per-channel quantization
        int64_t num_channels = float_tensor.size(axis);
        if (num_channels <= 0 || num_channels > 1024) {
            return 0;  // Skip unreasonable sizes
        }
        
        // Create scales (must be positive)
        std::vector<double> scales_vec(num_channels);
        for (int64_t i = 0; i < num_channels; i++) {
            if (offset < Size) {
                // Scale must be positive, use value between 0.01 and 2.55
                scales_vec[i] = 0.01 + (static_cast<double>(Data[offset++]) / 100.0);
            } else {
                scales_vec[i] = 0.1;
            }
        }
        torch::Tensor scales = torch::tensor(scales_vec, torch::kFloat64);
        
        // Create zero_points for quantization
        std::vector<int64_t> zp_vec(num_channels);
        for (int64_t i = 0; i < num_channels; i++) {
            if (offset < Size) {
                // Zero point typically in range [-128, 127] for int8
                zp_vec[i] = static_cast<int64_t>(Data[offset++]) - 128;
            } else {
                zp_vec[i] = 0;
            }
        }
        torch::Tensor zero_points_input = torch::tensor(zp_vec, torch::kInt64);
        
        // Create per-channel quantized tensor
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_channel(
                float_tensor, 
                scales, 
                zero_points_input, 
                axis, 
                torch::kQInt8
            );
        } catch (const c10::Error& e) {
            // Quantization may fail for certain tensor configurations
            return 0;
        } catch (const std::exception& e) {
            // Other exceptions during quantization setup
            return 0;
        }
        
        // Now test the actual API: q_per_channel_zero_points
        torch::Tensor retrieved_zero_points = quantized_tensor.q_per_channel_zero_points();
        
        // Verify the retrieved zero points
        if (retrieved_zero_points.defined()) {
            // Check properties
            auto dtype = retrieved_zero_points.dtype();
            auto numel = retrieved_zero_points.numel();
            auto sizes = retrieved_zero_points.sizes();
            
            // Verify the number of zero points matches the number of channels
            if (numel != num_channels) {
                std::cerr << "Zero points count mismatch!" << std::endl;
            }
            
            // Access the data to ensure it's valid
            if (numel > 0 && numel <= 1024) {
                auto accessor = retrieved_zero_points.accessor<int64_t, 1>();
                for (int64_t i = 0; i < numel; i++) {
                    volatile int64_t val = accessor[i];
                    (void)val;
                }
            }
        }
        
        // Also test the free function version if available
        try {
            torch::Tensor zp_free_func = torch::q_per_channel_zero_points(quantized_tensor);
            if (zp_free_func.defined()) {
                volatile auto n = zp_free_func.numel();
                (void)n;
            }
        } catch (...) {
            // Free function version may not exist or may have different behavior
        }
        
        // Test with different quantization dtype (QUInt8)
        if (offset + num_channels < Size) {
            std::vector<int64_t> zp_vec2(num_channels);
            for (int64_t i = 0; i < num_channels; i++) {
                // Zero point for uint8 typically in range [0, 255]
                zp_vec2[i] = static_cast<int64_t>(Data[offset++]);
            }
            torch::Tensor zero_points_input2 = torch::tensor(zp_vec2, torch::kInt64);
            
            try {
                torch::Tensor quantized_tensor2 = torch::quantize_per_channel(
                    float_tensor, 
                    scales, 
                    zero_points_input2, 
                    axis, 
                    torch::kQUInt8
                );
                
                torch::Tensor retrieved_zp2 = quantized_tensor2.q_per_channel_zero_points();
                if (retrieved_zp2.defined()) {
                    volatile auto n = retrieved_zp2.numel();
                    (void)n;
                }
            } catch (...) {
                // May fail for certain configurations
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