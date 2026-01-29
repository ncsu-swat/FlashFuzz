#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Extract dimensions for matrix (need at least 2D for matrix_rank)
        uint8_t rows = (Data[offset++] % 16) + 1;  // 1-16 rows
        uint8_t cols = (Data[offset++] % 16) + 1;  // 1-16 cols
        
        // Get tolerance parameters
        double tol = 1e-5;
        if (offset < Size) {
            uint8_t tol_byte = Data[offset++];
            tol = std::pow(10.0, -10.0 + (tol_byte % 10));
        }
        
        // Get hermitian parameter
        bool hermitian = false;
        if (offset < Size) {
            hermitian = (Data[offset++] % 2) == 1;
        }
        
        // Create a 2D matrix tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // Ensure we have a floating point type for matrix_rank
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat32);
        }
        
        // Reshape to 2D matrix
        int64_t total_elements = input.numel();
        if (total_elements < 1) {
            return 0;
        }
        
        // Create matrix dimensions that fit the elements
        int64_t m = std::min(static_cast<int64_t>(rows), total_elements);
        int64_t n = total_elements / m;
        if (n < 1) n = 1;
        
        // Resize tensor to fit m*n elements and reshape to 2D
        input = input.flatten().slice(0, 0, m * n).reshape({m, n});
        
        // Test basic matrix_rank call
        try {
            torch::Tensor result = torch::linalg_matrix_rank(input);
        } catch (const c10::Error&) {
            // Expected for some invalid inputs
        }
        
        // Test with tolerance (using optional atol parameter)
        try {
            torch::Tensor result = torch::linalg_matrix_rank(input, tol, c10::nullopt, hermitian);
        } catch (const c10::Error&) {
            // Expected for some invalid inputs
        }
        
        // Test with hermitian flag (requires square matrix for hermitian=true)
        if (hermitian && m == n) {
            try {
                // For hermitian, make the matrix symmetric
                torch::Tensor symmetric = (input + input.transpose(-2, -1)) / 2.0;
                torch::Tensor result = torch::linalg_matrix_rank(symmetric, c10::nullopt, c10::nullopt, true);
            } catch (const c10::Error&) {
                // Expected for some invalid inputs
            }
        } else {
            try {
                torch::Tensor result = torch::linalg_matrix_rank(input, c10::nullopt, c10::nullopt, false);
            } catch (const c10::Error&) {
                // Expected for some invalid inputs
            }
        }
        
        // Test with batched input if we have enough data
        if (Size - offset > 16) {
            uint8_t batch_size = (Data[offset % Size] % 4) + 1;  // 1-4 batches
            
            try {
                torch::Tensor batched = input.unsqueeze(0).expand({batch_size, m, n}).clone();
                torch::Tensor result = torch::linalg_matrix_rank(batched);
            } catch (const c10::Error&) {
                // Expected for some invalid inputs
            }
        }
        
        // Test with different dtypes
        try {
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor result = torch::linalg_matrix_rank(double_input);
        } catch (const c10::Error&) {
            // Expected for some invalid inputs
        }
        
        // Test with rtol parameter
        try {
            torch::Tensor result = torch::linalg_matrix_rank(input, c10::nullopt, tol, false);
        } catch (const c10::Error&) {
            // Expected for some invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}