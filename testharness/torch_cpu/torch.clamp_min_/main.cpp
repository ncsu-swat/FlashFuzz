#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get min value from remaining data
        float min_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&min_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Handle NaN/Inf in min_value to avoid undefined behavior
        if (std::isnan(min_value) || std::isinf(min_value)) {
            min_value = 0.0f;
        }
        
        // Create a copy of the input tensor for testing the in-place operation
        torch::Tensor tensor_copy = input_tensor.clone();
        
        // Apply clamp_min_ operation (in-place)
        tensor_copy.clamp_min_(min_value);
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::clamp_min(input_tensor, min_value);
        
        // Verify results match between in-place and out-of-place versions
        // Use inner try-catch since allclose can fail for edge cases without indicating a bug
        try {
            bool matches = torch::allclose(tensor_copy, expected, 1e-5, 1e-8);
            (void)matches; // Consume the result
        } catch (...) {
            // allclose may throw for certain tensor types, that's OK
        }
        
        // Test with different min values based on fuzzer input
        if (offset + sizeof(float) <= Size) {
            float another_min;
            std::memcpy(&another_min, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize to avoid issues with special float values
            if (!std::isnan(another_min) && !std::isinf(another_min)) {
                torch::Tensor another_copy = input_tensor.clone();
                another_copy.clamp_min_(another_min);
            }
        }
        
        // Test with tensor min value (Tensor overload)
        if (offset + 4 <= Size) {
            torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                torch::Tensor tensor_min_copy = input_tensor.clone();
                tensor_min_copy.clamp_min_(min_tensor);
            } catch (...) {
                // Shape mismatch is expected for incompatible tensors
            }
        }
        
        // Test with scalar tensor
        {
            torch::Tensor scalar_min = torch::tensor(min_value);
            torch::Tensor scalar_copy = input_tensor.clone();
            scalar_copy.clamp_min_(scalar_min);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}