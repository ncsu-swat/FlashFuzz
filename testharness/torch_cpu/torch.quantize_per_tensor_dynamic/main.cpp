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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract quantization parameters from the remaining data
        int64_t dtype_int = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dtype_int, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Map the dtype_int to a valid ScalarType for quantized tensors
        // Only qint8 and quint8 are typically supported for quantization
        torch::ScalarType q_dtype;
        if (dtype_int % 2 == 0) {
            q_dtype = torch::kQInt8;
        } else {
            q_dtype = torch::kQUInt8;
        }
        
        // Extract reduce_range parameter (boolean)
        bool reduce_range = false;
        if (offset < Size) {
            reduce_range = (Data[offset++] & 0x01) != 0;
        }
        
        // Apply quantize_per_tensor_dynamic operation
        torch::Tensor quantized_tensor;
        
        // Ensure input tensor has a valid floating-point dtype for quantization
        if (input_tensor.scalar_type() != torch::kFloat && 
            input_tensor.scalar_type() != torch::kDouble && 
            input_tensor.scalar_type() != torch::kHalf) {
            // Convert to float if not already a floating-point type
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Apply the quantize_per_tensor_dynamic operation
        quantized_tensor = torch::quantize_per_tensor_dynamic(input_tensor, q_dtype, reduce_range);
        
        // Perform some operations on the quantized tensor to ensure it's valid
        auto dequantized = quantized_tensor.dequantize();
        
        // Access tensor properties to ensure they're valid
        auto sizes = quantized_tensor.sizes();
        auto numel = quantized_tensor.numel();
        auto dtype = quantized_tensor.dtype();
        
        // Try to perform operations that might trigger issues
        if (numel > 0) {
            auto q_scale = quantized_tensor.q_scale();
            auto q_zero_point = quantized_tensor.q_zero_point();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
