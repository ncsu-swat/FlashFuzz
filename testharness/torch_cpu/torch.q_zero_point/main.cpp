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
        
        // Create a quantized tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure the tensor is quantized
        // We need to create a quantized tensor for q_zero_point to work
        if (!tensor.is_quantized()) {
            // Create a quantized tensor with scale=0.1 and zero_point=10
            // Using per_tensor_affine as the most common quantization scheme
            double scale = 0.1;
            int64_t zero_point = 10;
            
            // Choose a quantized dtype based on the original tensor's dtype
            torch::ScalarType qtype;
            if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                // For floating point tensors, quantize to quint8
                qtype = torch::kQUInt8;
            } else if (tensor.dtype() == torch::kInt8 || tensor.dtype() == torch::kInt16 || 
                       tensor.dtype() == torch::kInt32 || tensor.dtype() == torch::kInt64) {
                // For integer tensors, quantize to qint8
                qtype = torch::kQInt8;
            } else {
                // Default to quint8 for other types
                qtype = torch::kQUInt8;
            }
            
            // Convert tensor to float first if needed
            if (tensor.dtype() != torch::kFloat && tensor.dtype() != torch::kDouble) {
                tensor = tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            tensor = torch::quantize_per_tensor(tensor, scale, zero_point, qtype);
        }
        
        // Apply the q_zero_point operation
        int64_t result = torch::q_zero_point(tensor);
        
        // Optionally test other variants or edge cases
        if (offset + 1 < Size) {
            // Try dequantizing and requantizing with different parameters
            auto dequantized = tensor.dequantize();
            
            // Use next byte to determine a new zero point
            int64_t new_zero_point = static_cast<int64_t>(Data[offset++]) % 256;
            
            // Get the scale from the original tensor
            double scale = tensor.q_scale();
            
            // Requantize with new zero point
            auto requantized = torch::quantize_per_tensor(dequantized, scale, new_zero_point, tensor.scalar_type());
            
            // Get zero point from requantized tensor
            int64_t zero_point_result = torch::q_zero_point(requantized);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}