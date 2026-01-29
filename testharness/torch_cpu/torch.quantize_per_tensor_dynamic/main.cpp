#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Ensure input tensor has a valid floating-point dtype for quantization
        if (input_tensor.scalar_type() != torch::kFloat && 
            input_tensor.scalar_type() != torch::kDouble && 
            input_tensor.scalar_type() != torch::kHalf) {
            // Convert to float if not already a floating-point type
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Make sure tensor is contiguous for quantization
        input_tensor = input_tensor.contiguous();
        
        // Apply the quantize_per_tensor_dynamic operation
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_tensor_dynamic(input_tensor, q_dtype, reduce_range);
        } catch (const c10::Error &e) {
            // Expected failures for certain input configurations (e.g., empty tensors)
            return 0;
        }
        
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
            
            // Additional coverage: verify round-trip quantization-dequantization
            (void)dequantized.sum();
        }
        
        // Test with different input scenarios to improve coverage
        if (offset < Size && (Data[offset] & 0x03) == 0) {
            // Test with a clone to exercise memory paths
            auto cloned_input = input_tensor.clone();
            try {
                auto quantized_clone = torch::quantize_per_tensor_dynamic(cloned_input, q_dtype, reduce_range);
                (void)quantized_clone.dequantize();
            } catch (const c10::Error &e) {
                // Silently handle expected errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}