#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::ceil requires floating-point tensors
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Apply ceil operation
        torch::Tensor result = torch::ceil(input);
        
        // Try inplace version
        torch::Tensor input_copy = input.clone();
        input_copy.ceil_();
        
        // Try with out parameter
        torch::Tensor out = torch::empty_like(input);
        torch::ceil_out(out, input);
        
        // Try with non-contiguous tensor if possible
        if (input.dim() > 1 && input.size(0) > 1) {
            torch::Tensor non_contiguous = input.transpose(0, input.dim() - 1);
            if (!non_contiguous.is_contiguous()) {
                torch::Tensor result_non_contiguous = torch::ceil(non_contiguous);
            }
        }
        
        // Try with different floating-point dtypes
        if (offset < Size && Size - offset > 0) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::ScalarType dtype;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kHalf; break;
                case 3: dtype = torch::kBFloat16; break;
                default: dtype = torch::kFloat; break;
            }
            
            // Only try conversion if the dtype is different
            if (dtype != input.scalar_type()) {
                try {
                    torch::Tensor converted = input.to(dtype);
                    torch::Tensor result_converted = torch::ceil(converted);
                } catch (const std::exception&) {
                    // Conversion might fail for some dtypes, that's okay
                }
            }
        }
        
        // Test with scalar values derived from fuzzer data
        if (offset < Size) {
            float scalar_val = static_cast<float>(Data[offset++]) / 10.0f - 12.8f;
            torch::Tensor scalar_tensor = torch::tensor(scalar_val);
            torch::Tensor scalar_result = torch::ceil(scalar_tensor);
        }
        
        // Test with negative values
        torch::Tensor negative_input = input * -1.0;
        torch::Tensor negative_result = torch::ceil(negative_input);
        
        // Test with values that have fractional parts
        torch::Tensor fractional_input = input + 0.5;
        torch::Tensor fractional_result = torch::ceil(fractional_input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}