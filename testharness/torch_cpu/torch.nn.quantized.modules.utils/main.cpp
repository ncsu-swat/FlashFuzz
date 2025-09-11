#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a quantized tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) + sizeof(int64_t) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scale is positive and zero_point is within valid range
        scale = std::abs(scale);
        if (scale < 1e-10) scale = 1e-10;
        
        // Try different quantization operations
        
        // 1. Test quantize per tensor
        try {
            auto qtensor = torch::quantize_per_tensor(tensor, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // Ignore exceptions
        }
        
        // 2. Test quantize per channel
        if (tensor.dim() > 0) {
            try {
                int64_t axis = 0;
                if (offset < Size) {
                    axis = static_cast<int64_t>(Data[offset++]) % std::max(1, static_cast<int>(tensor.dim()));
                }
                auto scales = torch::ones({tensor.size(axis)}, torch::kFloat) * scale;
                auto zero_points = torch::zeros({tensor.size(axis)}, torch::kLong) + zero_point;
                auto qtensor = torch::quantize_per_channel(tensor, scales, zero_points, axis, torch::kQUInt8);
            } catch (...) {
                // Ignore exceptions
            }
        }
        
        // 3. Test dequantize
        try {
            auto qtensor = torch::quantize_per_tensor(tensor, scale, zero_point, torch::kQUInt8);
            auto dequant = qtensor.dequantize();
        } catch (...) {
            // Ignore exceptions
        }
        
        // 4. Test fake quantize per tensor affine
        try {
            auto fake_quant = torch::fake_quantize_per_tensor_affine(tensor, scale, zero_point, 0, 255);
        } catch (...) {
            // Ignore exceptions
        }
        
        // 5. Test fake quantize per channel affine
        if (tensor.dim() > 0) {
            try {
                int64_t axis = 0;
                if (offset < Size) {
                    axis = static_cast<int64_t>(Data[offset++]) % std::max(1, static_cast<int>(tensor.dim()));
                }
                auto scales = torch::ones({tensor.size(axis)}, torch::kFloat) * scale;
                auto zero_points = torch::zeros({tensor.size(axis)}, torch::kLong) + zero_point;
                auto fake_quant = torch::fake_quantize_per_channel_affine(tensor, scales, zero_points, axis, 0, 255);
            } catch (...) {
                // Ignore exceptions
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
