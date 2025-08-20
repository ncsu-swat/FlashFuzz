#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to dequantize
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to quantize the tensor first to ensure we have a quantized tensor
        // We'll use per_tensor quantization with different scale/zero_point values
        
        // Extract scale and zero_point from the remaining data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too extreme
            scale = std::abs(scale);
            if (scale < 1e-6f) scale = 1e-6f;
            if (scale > 1e6f) scale = 1e6f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for quantization
            zero_point = zero_point % 256;
        }
        
        // Try different quantization types based on remaining data
        uint8_t quant_type = 0;
        if (offset < Size) {
            quant_type = Data[offset++];
        }
        
        torch::Tensor quantized_tensor;
        
        // Try to quantize the tensor with different dtypes
        try {
            switch (quant_type % 3) {
                case 0:
                    // Per-tensor quantization with qint8
                    quantized_tensor = torch::quantize_per_tensor(tensor, scale, zero_point, torch::kQInt8);
                    break;
                case 1:
                    // Per-tensor quantization with quint8
                    quantized_tensor = torch::quantize_per_tensor(tensor, scale, zero_point, torch::kQUInt8);
                    break;
                case 2:
                    // Per-tensor quantization with qint32
                    quantized_tensor = torch::quantize_per_tensor(tensor, scale, zero_point, torch::kQInt32);
                    break;
            }
        } catch (const std::exception& e) {
            // If quantization fails, create a pre-quantized tensor for testing
            // This allows us to test dequantize even if quantize fails
            auto options = torch::TensorOptions().dtype(torch::kQInt8);
            quantized_tensor = torch::_empty_affine_quantized({2, 2}, options, scale, zero_point);
        }
        
        // Now dequantize the tensor
        torch::Tensor dequantized_tensor = torch::dequantize(quantized_tensor);
        
        // Try to access some properties of the dequantized tensor to ensure it's valid
        auto sizes = dequantized_tensor.sizes();
        auto dtype = dequantized_tensor.dtype();
        
        // Try some operations on the dequantized tensor
        if (dequantized_tensor.numel() > 0) {
            torch::Tensor result = dequantized_tensor + 1.0;
            result = result * 2.0;
        }
        
        // Try to dequantize a non-quantized tensor (should throw an exception)
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                torch::Tensor invalid_dequantize = torch::dequantize(tensor);
            } catch (const std::exception& e) {
                // Expected exception, continue
            }
        }
        
        // Try to dequantize the already dequantized tensor (should throw an exception)
        if (offset < Size && Data[offset] % 3 == 0) {
            try {
                torch::Tensor double_dequantize = torch::dequantize(dequantized_tensor);
            } catch (const std::exception& e) {
                // Expected exception, continue
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