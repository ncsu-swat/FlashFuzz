#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least some data to work with
        if (Size < 8) {
            return 0;
        }
        
        // Extract dimensions for vec1 and vec2 from fuzzer data
        int64_t dim1 = 1 + (Data[offset] % 32);  // 1-32
        offset++;
        int64_t dim2 = 1 + (Data[offset] % 32);  // 1-32
        offset++;
        
        // Create vec1 (1D tensor of size dim1)
        torch::Tensor vec1;
        if (offset < Size) {
            vec1 = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure vec1 is 1D
            vec1 = vec1.flatten();
            if (vec1.numel() == 0) {
                vec1 = torch::randn({dim1});
            } else if (vec1.numel() < dim1) {
                vec1 = torch::randn({dim1});
            } else {
                vec1 = vec1.slice(0, 0, dim1);
            }
        } else {
            vec1 = torch::randn({dim1});
        }
        
        // Create vec2 (1D tensor of size dim2)
        torch::Tensor vec2;
        if (offset < Size) {
            vec2 = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure vec2 is 1D
            vec2 = vec2.flatten();
            if (vec2.numel() == 0) {
                vec2 = torch::randn({dim2});
            } else if (vec2.numel() < dim2) {
                vec2 = torch::randn({dim2});
            } else {
                vec2 = vec2.slice(0, 0, dim2);
            }
        } else {
            vec2 = torch::randn({dim2});
        }
        
        // Create input tensor with compatible shape (dim1 x dim2)
        torch::Tensor input = torch::randn({vec1.size(0), vec2.size(0)}, vec1.options());
        
        // Get alpha and beta values from the input data if available
        float alpha = 1.0f;
        float beta = 1.0f;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp to reasonable range to avoid numerical issues
            if (!std::isfinite(alpha)) alpha = 1.0f;
            alpha = std::max(-100.0f, std::min(100.0f, alpha));
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isfinite(beta)) beta = 1.0f;
            beta = std::max(-100.0f, std::min(100.0f, beta));
        }
        
        // Basic addr operation: beta * input + alpha * (vec1 outer vec2)
        try {
            torch::Tensor result1 = torch::addr(input, vec1, vec2);
            (void)result1;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // addr with alpha and beta using Scalar
        try {
            torch::Tensor result2 = torch::addr(input, vec1, vec2, 
                                                 at::Scalar(beta), at::Scalar(alpha));
            (void)result2;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // addr_out variant
        try {
            torch::Tensor out = torch::zeros({vec1.size(0), vec2.size(0)}, vec1.options());
            torch::addr_out(out, input, vec1, vec2);
            (void)out;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // addr_out with alpha and beta
        try {
            torch::Tensor out2 = torch::zeros({vec1.size(0), vec2.size(0)}, vec1.options());
            torch::addr_out(out2, input, vec1, vec2, at::Scalar(beta), at::Scalar(alpha));
            (void)out2;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // In-place variant
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.addr_(vec1, vec2);
            (void)input_copy;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // In-place variant with alpha and beta
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.addr_(vec1, vec2, at::Scalar(beta), at::Scalar(alpha));
            (void)input_copy;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // Test with different dtypes
        try {
            torch::Tensor vec1_double = vec1.to(torch::kDouble);
            torch::Tensor vec2_double = vec2.to(torch::kDouble);
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::Tensor result = torch::addr(input_double, vec1_double, vec2_double);
            (void)result;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // Test with zero-valued scalars
        try {
            torch::Tensor result = torch::addr(input, vec1, vec2, 
                                               at::Scalar(0.0), at::Scalar(1.0));
            (void)result;
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}