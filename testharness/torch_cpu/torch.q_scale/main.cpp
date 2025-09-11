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
        if (Size < 2) {
            return 0;
        }
        
        // Create a quantized tensor to test q_scale
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a quantized tensor if the input tensor is not already quantized
        torch::Tensor quantized_tensor;
        if (!tensor.is_quantized()) {
            // Convert to quantized tensor if not already quantized
            // We'll use per_tensor quantization for simplicity
            double scale = 0.1;
            int zero_point = 0;
            
            // Use remaining data to determine scale if available
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scale, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Ensure scale is positive and not too small or large
                scale = std::abs(scale);
                if (scale < 1e-10) scale = 1e-10;
                if (scale > 1e10) scale = 1e10;
            }
            
            // Use remaining data to determine zero_point if available
            if (offset + sizeof(int) <= Size) {
                std::memcpy(&zero_point, Data + offset, sizeof(int));
                offset += sizeof(int);
            }
            
            // Quantize the tensor - use qint8 as the target type
            quantized_tensor = torch::quantize_per_tensor(
                tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQInt8
            );
        } else {
            quantized_tensor = tensor;
        }
        
        // Apply q_scale operation to get the scale of the quantized tensor
        double scale_value = torch::q_scale(quantized_tensor);
        
        // Try to use the result to ensure it's not optimized away
        if (std::isnan(scale_value) || std::isinf(scale_value)) {
            // Just a check to handle extreme values, but we don't throw
            // to allow the fuzzer to continue
        }
        
        // Test q_scale on a view of the tensor if possible
        if (quantized_tensor.dim() > 0 && quantized_tensor.size(0) > 1) {
            torch::Tensor view = quantized_tensor.slice(0, 0, quantized_tensor.size(0) / 2);
            double view_scale = torch::q_scale(view);
        }
        
        // Test q_scale on a reshaped tensor if possible
        if (quantized_tensor.numel() > 1) {
            std::vector<int64_t> new_shape;
            if (quantized_tensor.dim() == 0) {
                new_shape = {1};
            } else if (quantized_tensor.dim() == 1) {
                new_shape = {1, quantized_tensor.size(0)};
            } else {
                // Flatten to 1D
                new_shape = {quantized_tensor.numel()};
            }
            
            torch::Tensor reshaped = quantized_tensor.reshape(new_shape);
            double reshaped_scale = torch::q_scale(reshaped);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
