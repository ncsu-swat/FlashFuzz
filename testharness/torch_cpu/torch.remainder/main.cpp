#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::min

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create divisor tensor
        torch::Tensor divisor;
        if (offset < Size) {
            divisor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a scalar tensor
            divisor = torch::tensor(1.0, input.options());
        }
        
        // Try different variants of remainder operation
        
        // 1. Tensor.remainder(other)
        try {
            torch::Tensor result1 = input.remainder(divisor);
        } catch (const std::exception&) {
            // Shape mismatch or type error - expected
        }
        
        // 2. torch::remainder(input, other)
        try {
            torch::Tensor result2 = torch::remainder(input, divisor);
        } catch (const std::exception&) {
            // Shape mismatch or type error - expected
        }
        
        // 3. torch::fmod(input, other) - similar operation to test
        try {
            torch::Tensor result3 = torch::fmod(input, divisor);
        } catch (const std::exception&) {
            // Shape mismatch or type error - expected
        }
        
        // 4. Test with scalar divisor
        if (offset < Size) {
            // Use a byte from the input data as a scalar (avoid zero)
            double scalar_value = static_cast<double>(Data[offset % Size]);
            if (scalar_value == 0.0) {
                scalar_value = 1.0;
            }
            
            try {
                torch::Tensor result4 = input.remainder(scalar_value);
            } catch (const std::exception&) {
                // Type error - expected for some dtypes
            }
            
            // 5. Test in-place version
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.remainder_(scalar_value);
            } catch (const std::exception&) {
                // Type error - expected for some dtypes
            }
        }
        
        // 6. Test with different scalar types (only floating point types supported)
        if (offset + 1 < Size) {
            uint8_t type_selector = Data[offset++] % 3;
            torch::ScalarType dtype;
            switch (type_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                default: dtype = torch::kFloat16; break;
            }
            
            try {
                // Convert input to the selected data type
                torch::Tensor typed_input = input.to(dtype);
                torch::Tensor typed_divisor = divisor.to(dtype);
                
                // Test remainder with the converted tensors
                torch::Tensor typed_result = torch::remainder(typed_input, typed_divisor);
            } catch (const std::exception&) {
                // Conversion or operation error - expected
            }
        }
        
        // 7. Test with zero divisor (should trigger division by zero handling)
        try {
            torch::Tensor zero_divisor = torch::zeros_like(divisor);
            torch::Tensor zero_result = torch::remainder(input, zero_divisor);
        } catch (const std::exception&) {
            // Expected exception for division by zero
        }
        
        // 8. Test with broadcasting
        if (input.dim() > 0 && divisor.dim() > 0) {
            try {
                // Create a smaller tensor for broadcasting
                std::vector<int64_t> smaller_shape;
                for (int i = 0; i < std::min(static_cast<int>(input.dim()), 2); i++) {
                    smaller_shape.push_back(1);
                }
                
                torch::Tensor broadcast_divisor = torch::ones(smaller_shape, divisor.options());
                torch::Tensor broadcast_result = torch::remainder(input, broadcast_divisor);
            } catch (const std::exception&) {
                // Broadcasting error - expected
            }
        }
        
        // 9. Test with negative values
        try {
            torch::Tensor neg_input = input.neg();
            torch::Tensor neg_result = torch::remainder(neg_input, divisor);
        } catch (const std::exception&) {
            // Type error - expected for some dtypes
        }
        
        // 10. Test remainder with integer types
        if (offset < Size) {
            try {
                torch::Tensor int_input = input.to(torch::kInt32);
                int32_t int_divisor = static_cast<int32_t>(Data[offset % Size]);
                if (int_divisor == 0) {
                    int_divisor = 1;
                }
                torch::Tensor int_result = int_input.remainder(int_divisor);
            } catch (const std::exception&) {
                // Type conversion error - expected
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