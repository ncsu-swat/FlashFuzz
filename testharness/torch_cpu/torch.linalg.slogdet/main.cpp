#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // slogdet requires a square matrix (n x n) or batch of square matrices
        // Reshape to a valid square matrix
        int64_t total_elements = input.numel();
        if (total_elements < 1) {
            return 0;
        }
        
        int64_t matrix_size = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
        if (matrix_size < 1) {
            matrix_size = 1;
        }
        
        // Flatten and take only the elements we need for a square matrix
        input = input.flatten().slice(0, 0, matrix_size * matrix_size).reshape({matrix_size, matrix_size});
        
        // Ensure float type for linalg operations
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply slogdet operation (torch::slogdet is available in C++ frontend)
        auto result = torch::slogdet(input);
        
        // Unpack the result (sign, logabsdet)
        auto sign = std::get<0>(result);
        auto logabsdet = std::get<1>(result);
        
        // Use results to prevent optimization
        volatile float sign_val = sign.item<float>();
        volatile float logabsdet_val = logabsdet.item<float>();
        (void)sign_val;
        (void)logabsdet_val;
        
        // Additional test cases based on fuzzer data
        if (offset < Size) {
            uint8_t op_selector = Data[offset++];
            
            switch (op_selector % 5) {
                case 0:
                    // Test with transposed matrix (should give same result)
                    {
                        auto transposed = input.transpose(0, 1).contiguous();
                        auto trans_result = torch::slogdet(transposed);
                    }
                    break;
                case 1:
                    // Test with scaled matrix
                    {
                        float scale = (offset < Size) ? (Data[offset++] / 128.0f + 0.1f) : 2.0f;
                        auto scaled = input * scale;
                        auto scaled_result = torch::slogdet(scaled);
                    }
                    break;
                case 2:
                    // Test with batched input (add batch dimension)
                    {
                        auto batched = input.unsqueeze(0).expand({2, matrix_size, matrix_size}).contiguous();
                        auto batch_result = torch::slogdet(batched);
                    }
                    break;
                case 3:
                    // Test with complex input
                    try {
                        auto complex_input = torch::complex(input, torch::zeros_like(input));
                        auto complex_result = torch::slogdet(complex_input);
                    } catch (...) {
                        // Complex may not be supported in all builds
                    }
                    break;
                case 4:
                    // Test with double precision
                    {
                        auto double_input = input.to(torch::kFloat64);
                        auto double_result = torch::slogdet(double_input);
                    }
                    break;
            }
        }
        
        // Test edge cases: identity matrix and zero matrix
        if (offset < Size && Data[offset] % 10 == 0) {
            auto identity = torch::eye(matrix_size, input.options());
            auto identity_result = torch::slogdet(identity);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}