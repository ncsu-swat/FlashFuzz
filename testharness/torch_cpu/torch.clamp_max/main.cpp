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
        
        // Need at least a few bytes for input tensor and max value
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract max value from remaining data
        double max_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Handle NaN and Inf to avoid undefined behavior
            if (std::isnan(max_value) || std::isinf(max_value)) {
                max_value = static_cast<double>(Data[offset > 0 ? offset - 1 : 0] % 200) - 100.0;
            }
        } else if (offset < Size) {
            // Use remaining bytes to create a scalar value
            uint8_t byte_value = Data[offset++];
            max_value = static_cast<double>(byte_value) - 128.0; // Range [-128, 127]
        }
        
        // Apply clamp_max in different ways
        
        // 1. Using torch::clamp_max directly
        torch::Tensor result1 = torch::clamp_max(input, max_value);
        
        // 2. Using the tensor's clamp_max method
        torch::Tensor result2 = input.clamp_max(max_value);
        
        // 3. Using out variant
        torch::Tensor out = torch::empty_like(input);
        torch::clamp_max_out(out, input, max_value);
        
        // 4. Using in-place variant
        torch::Tensor input_copy = input.clone();
        input_copy.clamp_max_(max_value);
        
        // 5. Try with a tensor max value if there's enough data left
        if (offset < Size) {
            try {
                torch::Tensor max_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Handle scalar tensor case
                if (max_tensor.dim() == 0) {
                    torch::Tensor result_tensor_max = torch::clamp_max(input, max_tensor);
                } else {
                    // Try element-wise max values - broadcasting may or may not work
                    torch::Tensor result_tensor_max = torch::clamp_max(input, max_tensor);
                }
            } catch (const std::exception &) {
                // Silently catch expected shape mismatch or type errors
            }
        }
        
        // 6. Try with different scalar types for max value
        torch::Tensor result_int_max = torch::clamp_max(input, static_cast<int64_t>(max_value));
        torch::Tensor result_float_max = torch::clamp_max(input, static_cast<float>(max_value));
        
        // 7. Edge case: Try with extreme but valid values
        torch::Tensor result_large_max = torch::clamp_max(input, 1e30);
        torch::Tensor result_small_max = torch::clamp_max(input, -1e30);
        
        // 8. Try clamp_max on different tensor types
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor result_float = torch::clamp_max(float_input, max_value);
            
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor result_double = torch::clamp_max(double_input, max_value);
            
            torch::Tensor int_input = input.to(torch::kInt32);
            torch::Tensor result_int = torch::clamp_max(int_input, static_cast<int64_t>(max_value));
        } catch (const std::exception &) {
            // Silently catch type conversion errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}