#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        
        // Create another tensor for the "other" parameter
        torch::Tensor other;
        if (offset + 4 <= Size) {
            other = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Use a scalar tensor if we don't have enough data
            other = torch::tensor(2.0, input.options());
        }
        
        // Get alpha value from remaining data if available
        double alpha = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize alpha to avoid NaN/Inf issues that aren't bugs
            if (!std::isfinite(alpha)) {
                alpha = 1.0;
            }
        }
        
        // 1. Basic rsub: computes other - input
        // torch::rsub(input, other) = other - input
        try {
            torch::Tensor result1 = torch::rsub(input, other);
            (void)result1;
        } catch (const std::exception &) {
            // Shape/dtype mismatches are expected
        }
        
        // 2. rsub with alpha: computes other - input * alpha
        try {
            torch::Tensor result2 = torch::rsub(input, other, alpha);
            (void)result2;
        } catch (const std::exception &) {
            // Expected for incompatible tensors
        }
        
        // 3. rsub with Scalar value
        try {
            double scalar_value = 5.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                if (!std::isfinite(scalar_value)) {
                    scalar_value = 5.0;
                }
            }
            // rsub with scalar: scalar - input
            torch::Tensor result3 = torch::rsub(input, torch::Scalar(scalar_value));
            (void)result3;
        } catch (const std::exception &) {
            // Expected for some dtypes
        }
        
        // 4. rsub with scalar and alpha
        try {
            double scalar_value = 3.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                if (!std::isfinite(scalar_value)) {
                    scalar_value = 3.0;
                }
            }
            torch::Tensor result4 = torch::rsub(input, torch::Scalar(scalar_value), torch::Scalar(alpha));
            (void)result4;
        } catch (const std::exception &) {
            // Expected
        }
        
        // 5. Test with different tensor types - broadcast scenario
        try {
            // Create a 1D tensor that can broadcast
            auto options = torch::TensorOptions().dtype(input.dtype());
            torch::Tensor broadcast_other = torch::ones({1}, options);
            torch::Tensor result5 = torch::rsub(input, broadcast_other);
            (void)result5;
        } catch (const std::exception &) {
            // Expected for some cases
        }
        
        // 6. Test with zero-dimensional tensor (scalar tensor)
        try {
            auto options = torch::TensorOptions().dtype(input.dtype());
            torch::Tensor scalar_tensor = torch::tensor(2.5, options);
            torch::Tensor result6 = torch::rsub(input, scalar_tensor, alpha);
            (void)result6;
        } catch (const std::exception &) {
            // Expected
        }
        
        // 7. Test with same-shape tensors
        try {
            torch::Tensor same_shape = torch::ones_like(input);
            torch::Tensor result7 = torch::rsub(input, same_shape);
            (void)result7;
        } catch (const std::exception &) {
            // Expected
        }
        
        // 8. Test with negative alpha
        try {
            torch::Tensor result8 = torch::rsub(input, other, -alpha);
            (void)result8;
        } catch (const std::exception &) {
            // Expected
        }
        
        // 9. Test rsub on integer tensors
        try {
            torch::Tensor int_input = input.to(torch::kInt32);
            torch::Tensor int_other = torch::ones_like(int_input) * 10;
            torch::Tensor result9 = torch::rsub(int_input, int_other);
            (void)result9;
        } catch (const std::exception &) {
            // Expected for some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}