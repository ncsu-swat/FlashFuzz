#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        
        // Need at least a few bytes for the input tensor and quantization parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // quantize_per_tensor requires float tensor input
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Ensure tensor is contiguous (required for quantization)
        input_tensor = input_tensor.contiguous();
        
        // Extract quantization parameters from the remaining data
        double scale = 1.0;
        int64_t zero_point = 0;
        torch::ScalarType dtype = torch::kQInt8;
        
        // Parse scale (ensure we have enough data)
        if (offset + sizeof(float) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and finite (required by quantize_per_tensor)
            if (!std::isfinite(scale_f) || scale_f <= 0) {
                scale = 1.0;
            } else {
                scale = static_cast<double>(scale_f);
            }
        }
        
        // Parse dtype for quantization first (needed to constrain zero_point)
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // Select between the supported quantized dtypes
            switch (dtype_selector % 3) {
                case 0:
                    dtype = torch::kQInt8;
                    break;
                case 1:
                    dtype = torch::kQUInt8;
                    break;
                case 2:
                    dtype = torch::kQInt32;
                    break;
            }
        }
        
        // Parse zero_point with appropriate range constraints
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Constrain zero_point based on dtype
            if (dtype == torch::kQUInt8) {
                zero_point = std::abs(zero_point) % 256;  // [0, 255]
            } else if (dtype == torch::kQInt8) {
                zero_point = (zero_point % 256) - 128;  // [-128, 127]
            }
            // kQInt32 has a much larger range, no constraint needed
        }
        
        // Apply quantize_per_tensor operation
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, dtype);
        } catch (const c10::Error &e) {
            // Expected failures for certain tensor configurations
            return 0;
        }
        
        // Dequantize to verify the operation completed successfully
        torch::Tensor dequantized_tensor = quantized_tensor.dequantize();
        
        // Access some values to ensure computation is not optimized away
        if (dequantized_tensor.numel() > 0) {
            volatile float first_val = dequantized_tensor.flatten()[0].item<float>();
            (void)first_val;
        }
        
        // Additional coverage: test q_scale() and q_zero_point() accessors
        volatile double q_scale = quantized_tensor.q_scale();
        volatile int64_t q_zp = quantized_tensor.q_zero_point();
        (void)q_scale;
        (void)q_zp;
        
        // Test int_repr() to get underlying integer representation
        torch::Tensor int_repr = quantized_tensor.int_repr();
        if (int_repr.numel() > 0) {
            volatile int first_int = int_repr.flatten()[0].item<int>();
            (void)first_int;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}