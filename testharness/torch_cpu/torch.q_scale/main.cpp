#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs, std::isnan, std::isinf

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
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to be quantized
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float for quantization
        torch::Tensor float_tensor = tensor.to(torch::kFloat);
        
        // Use remaining data to determine scale if available
        double scale = 0.1;
        if (offset + sizeof(float) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and in reasonable range
            scale = static_cast<double>(std::abs(scale_f));
            if (scale < 1e-6) scale = 1e-6;
            if (scale > 1e6) scale = 1e6;
            if (std::isnan(scale) || std::isinf(scale)) scale = 0.1;
        }
        
        // Use remaining data to determine zero_point if available
        int64_t zero_point = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t zp;
            std::memcpy(&zp, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            zero_point = static_cast<int64_t>(zp);  // Already in valid range for qint8
        }
        
        // Quantize the tensor using per_tensor quantization
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_tensor(
                float_tensor, 
                scale, 
                zero_point, 
                torch::kQInt8
            );
        } catch (const std::exception &e) {
            // Quantization can fail for various reasons, silently skip
            return 0;
        }
        
        // Apply q_scale operation to get the scale of the quantized tensor
        double scale_value = quantized_tensor.q_scale();
        
        // Verify the scale value is valid
        if (std::isnan(scale_value) || std::isinf(scale_value)) {
            // Log unexpected invalid scale values
            std::cerr << "Warning: q_scale returned invalid value" << std::endl;
        }
        
        // Also test the free function form
        double scale_value2 = torch::q_scale(quantized_tensor);
        (void)scale_value;
        (void)scale_value2;
        
        // Test q_scale on contiguous copy
        torch::Tensor contiguous = quantized_tensor.contiguous();
        double contiguous_scale = torch::q_scale(contiguous);
        (void)contiguous_scale;
        
        // Test q_scale after clone
        torch::Tensor cloned = quantized_tensor.clone();
        double cloned_scale = torch::q_scale(cloned);
        (void)cloned_scale;
        
        // Test with different quantization types if we have more data
        if (offset < Size) {
            uint8_t quant_type = Data[offset] % 2;
            offset++;
            
            try {
                torch::Tensor quantized_tensor2;
                if (quant_type == 0) {
                    // Use kQUInt8
                    int64_t zp_uint8 = std::abs(zero_point) % 256;  // 0-255 for quint8
                    quantized_tensor2 = torch::quantize_per_tensor(
                        float_tensor, 
                        scale, 
                        zp_uint8, 
                        torch::kQUInt8
                    );
                } else {
                    // Use kQInt32
                    quantized_tensor2 = torch::quantize_per_tensor(
                        float_tensor, 
                        scale, 
                        zero_point, 
                        torch::kQInt32
                    );
                }
                
                double scale_value3 = torch::q_scale(quantized_tensor2);
                (void)scale_value3;
            } catch (const std::exception &e) {
                // Different quantization types may fail, silently skip
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}