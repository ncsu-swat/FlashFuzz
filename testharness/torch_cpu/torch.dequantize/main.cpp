#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to dequantize
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float for quantization
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Extract scale and zero_point from the remaining data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too extreme
            scale = std::abs(scale);
            if (!std::isfinite(scale) || scale < 1e-6f) scale = 1e-6f;
            if (scale > 1e6f) scale = 1e6f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try different quantization types based on remaining data
        uint8_t quant_type = 0;
        if (offset < Size) {
            quant_type = Data[offset++];
        }
        
        torch::Tensor quantized_tensor;
        bool quantization_succeeded = false;
        
        // Try to quantize the tensor with different dtypes
        try {
            switch (quant_type % 3) {
                case 0: {
                    // Per-tensor quantization with qint8
                    // zero_point for qint8 should be in [-128, 127]
                    int64_t zp = (zero_point % 256) - 128;
                    quantized_tensor = torch::quantize_per_tensor(tensor, scale, zp, torch::kQInt8);
                    quantization_succeeded = true;
                    break;
                }
                case 1: {
                    // Per-tensor quantization with quint8
                    // zero_point for quint8 should be in [0, 255]
                    int64_t zp = std::abs(zero_point) % 256;
                    quantized_tensor = torch::quantize_per_tensor(tensor, scale, zp, torch::kQUInt8);
                    quantization_succeeded = true;
                    break;
                }
                case 2: {
                    // Per-tensor quantization with qint32
                    // zero_point for qint32 must be 0
                    quantized_tensor = torch::quantize_per_tensor(tensor, scale, 0, torch::kQInt32);
                    quantization_succeeded = true;
                    break;
                }
            }
        } catch (const std::exception& e) {
            // Quantization failed, try to create a simple quantized tensor
        }
        
        if (!quantization_succeeded) {
            // Create a simple float tensor and quantize it
            try {
                auto simple_tensor = torch::randn({2, 2});
                quantized_tensor = torch::quantize_per_tensor(simple_tensor, 0.1, 0, torch::kQUInt8);
                quantization_succeeded = true;
            } catch (const std::exception& e) {
                // If we still can't create a quantized tensor, skip this input
                return 0;
            }
        }
        
        // Now dequantize the tensor
        torch::Tensor dequantized_tensor = torch::dequantize(quantized_tensor);
        
        // Verify the dequantized tensor is valid and non-quantized
        auto sizes = dequantized_tensor.sizes();
        auto dtype = dequantized_tensor.dtype();
        
        // Try some operations on the dequantized tensor
        if (dequantized_tensor.numel() > 0) {
            torch::Tensor result = dequantized_tensor + 1.0;
            result = result * 2.0;
            // Compute sum to exercise more code paths
            auto sum = result.sum();
        }
        
        // Test dequantize on a list of quantized tensors
        if (offset < Size && Data[offset] % 4 == 0) {
            try {
                std::vector<torch::Tensor> quantized_list = {quantized_tensor};
                if (quantization_succeeded) {
                    // Add another quantized tensor to the list
                    auto another_tensor = torch::randn_like(dequantized_tensor);
                    auto another_quantized = torch::quantize_per_tensor(another_tensor, scale, 0, torch::kQUInt8);
                    quantized_list.push_back(another_quantized);
                }
                auto dequantized_list = torch::dequantize(quantized_list);
            } catch (const std::exception& e) {
                // Expected - some configurations may fail
            }
        }
        
        // Try to dequantize a non-quantized tensor (should throw an exception)
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                torch::Tensor invalid_dequantize = torch::dequantize(tensor);
            } catch (const std::exception& e) {
                // Expected exception for non-quantized tensor
            }
        }
        
        // Try to dequantize the already dequantized tensor (should throw an exception)
        if (offset + 1 < Size && Data[offset + 1] % 3 == 0) {
            try {
                torch::Tensor double_dequantize = torch::dequantize(dequantized_tensor);
            } catch (const std::exception& e) {
                // Expected exception
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}