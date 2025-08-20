#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor with various properties
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the quantization scheme of the tensor
        auto qscheme = tensor.qscheme();
        
        // Try to create a quantized tensor to test qscheme more thoroughly
        if (offset + 2 < Size) {
            // Extract a scale factor from the data
            float scale = 0.01f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Ensure scale is positive and reasonable
                scale = std::abs(scale);
                if (scale < 1e-6f) scale = 1e-6f;
                if (scale > 1e6f) scale = 1e6f;
            }
            
            // Extract a zero point from the data
            int32_t zero_point = 0;
            if (offset + sizeof(int32_t) <= Size) {
                std::memcpy(&zero_point, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            }
            
            // Try different quantization schemes
            uint8_t scheme_selector = 0;
            if (offset < Size) {
                scheme_selector = Data[offset++];
            }
            
            // Only attempt quantization on floating point tensors
            if (tensor.is_floating_point()) {
                try {
                    torch::Tensor quantized_tensor;
                    
                    // Select different quantization schemes based on input data
                    switch (scheme_selector % 4) {
                        case 0:
                            // Per tensor quantization
                            quantized_tensor = torch::quantize_per_tensor(
                                tensor, scale, zero_point, torch::kQInt8);
                            break;
                        case 1:
                            // Per tensor quantization with different dtype
                            quantized_tensor = torch::quantize_per_tensor(
                                tensor, scale, zero_point, torch::kQUInt8);
                            break;
                        case 2:
                            // Try per channel quantization if tensor has at least 1 dimension
                            if (tensor.dim() > 0) {
                                int64_t axis = 0;
                                if (tensor.dim() > 1) {
                                    // Use a different axis if available
                                    axis = scheme_selector % tensor.dim();
                                }
                                
                                // Create scales and zero points for each channel
                                auto num_channels = tensor.size(axis);
                                auto scales = torch::ones(num_channels) * scale;
                                auto zero_points = torch::zeros(num_channels, torch::kInt);
                                
                                quantized_tensor = torch::quantize_per_channel(
                                    tensor, scales, zero_points, axis, torch::kQInt8);
                            } else {
                                // Fallback for scalar tensors
                                quantized_tensor = torch::quantize_per_tensor(
                                    tensor, scale, zero_point, torch::kQInt8);
                            }
                            break;
                        case 3:
                            // Try per channel quantization with different dtype
                            if (tensor.dim() > 0) {
                                int64_t axis = 0;
                                if (tensor.dim() > 1) {
                                    axis = scheme_selector % tensor.dim();
                                }
                                
                                auto num_channels = tensor.size(axis);
                                auto scales = torch::ones(num_channels) * scale;
                                auto zero_points = torch::zeros(num_channels, torch::kInt);
                                
                                quantized_tensor = torch::quantize_per_channel(
                                    tensor, scales, zero_points, axis, torch::kQInt32);
                            } else {
                                quantized_tensor = torch::quantize_per_tensor(
                                    tensor, scale, zero_point, torch::kQInt32);
                            }
                            break;
                    }
                    
                    // Get the qscheme of the quantized tensor
                    auto quantized_qscheme = quantized_tensor.qscheme();
                    
                    // Test dequantization
                    auto dequantized = quantized_tensor.dequantize();
                    
                    // Get the qscheme of the dequantized tensor
                    auto dequantized_qscheme = dequantized.qscheme();
                } catch (const std::exception& e) {
                    // Quantization might fail for various reasons, that's fine for fuzzing
                }
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