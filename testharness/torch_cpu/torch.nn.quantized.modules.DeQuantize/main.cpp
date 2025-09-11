#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a quantized tensor to dequantize
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        // Use remaining data to influence scale and zero_point if available
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too extreme
            scale = std::abs(scale);
            if (scale < 1e-10f) scale = 1e-10f;
            if (scale > 1e10f) scale = 1e10f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for int8
            zero_point = std::max(std::min(zero_point, static_cast<int64_t>(127)), static_cast<int64_t>(-128));
        }
        
        // Quantize the tensor first (we need a quantized tensor to dequantize)
        torch::Tensor quantized;
        
        // Try different quantization types based on input data
        if (offset < Size) {
            uint8_t quant_type = Data[offset++];
            
            if (quant_type % 3 == 0) {
                // Per tensor quantization
                quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
            } else if (quant_type % 3 == 1) {
                // Per channel quantization (if tensor has enough dimensions)
                if (input_tensor.dim() > 0) {
                    int64_t axis = 0;
                    if (input_tensor.dim() > 1) {
                        axis = Data[offset % Size] % input_tensor.dim();
                    }
                    
                    // Create scales and zero_points for each channel
                    int64_t num_channels = input_tensor.size(axis);
                    std::vector<double> scales(num_channels, scale);
                    std::vector<int64_t> zero_points(num_channels, zero_point);
                    
                    quantized = torch::quantize_per_channel(
                        input_tensor, 
                        torch::from_blob(scales.data(), {num_channels}, torch::kDouble).clone(),
                        torch::from_blob(zero_points.data(), {num_channels}, torch::kLong).clone(),
                        axis,
                        torch::kQInt8
                    );
                } else {
                    // Fallback to per tensor for scalars
                    quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
                }
            } else {
                // Try with different dtype
                quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
            }
        } else {
            // Default quantization
            quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
        }
        
        // Apply dequantization using torch::dequantize function
        torch::Tensor dequantized = torch::dequantize(quantized);
        
        // Verify the dequantized tensor has the same shape as the original
        if (dequantized.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("Dequantized tensor has different shape than input");
        }
        
        // Try some operations on the dequantized tensor to ensure it's valid
        torch::Tensor result = dequantized + 1.0;
        result = result * 2.0;
        result = torch::relu(result);
        
        // Access some values to ensure tensor is valid
        if (result.numel() > 0) {
            float first_val = result.item<float>();
            (void)first_val; // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
