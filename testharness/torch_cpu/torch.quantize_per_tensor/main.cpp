#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and quantization parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract quantization parameters from the remaining data
        double scale = 1.0;
        int64_t zero_point = 0;
        torch::ScalarType dtype = torch::kQInt8;
        
        // Parse scale (ensure we have enough data)
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive (required by quantize_per_tensor)
            if (scale <= 0) {
                scale = std::abs(scale);
                if (scale == 0) scale = 1.0;
            }
        }
        
        // Parse zero_point
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dtype for quantization
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // Select between the supported quantized dtypes
            if (dtype_selector % 3 == 0) {
                dtype = torch::kQInt8;
            } else if (dtype_selector % 3 == 1) {
                dtype = torch::kQUInt8;
            } else {
                dtype = torch::kQInt32;
            }
        }
        
        // Apply quantize_per_tensor operation
        torch::Tensor quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, dtype);
        
        // Dequantize to verify the operation completed successfully
        torch::Tensor dequantized_tensor = quantized_tensor.dequantize();
        
        // Access some values to ensure computation is not optimized away
        if (dequantized_tensor.numel() > 0) {
            volatile float first_val = dequantized_tensor.item<float>();
            (void)first_val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}