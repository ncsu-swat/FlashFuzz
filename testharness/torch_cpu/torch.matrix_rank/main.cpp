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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract tolerance parameter from remaining data if available
        double tol = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&tol, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure tolerance is valid (positive, not NaN/Inf)
            if (!std::isfinite(tol) || tol < 0) {
                tol = 1e-5;
            }
            if (tol < 1e-10) tol = 1e-10;
            if (tol > 1.0) tol = 1.0;
        }
        
        // Extract boolean parameter for hermitian flag if available
        bool hermitian = false;
        if (offset < Size) {
            hermitian = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // matrix_rank requires at least 2D tensor
        if (input.dim() < 2) {
            // Reshape to 2D
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0;
            }
            input = input.reshape({1, numel});
        }
        
        // Convert tensor to float if it's an integer type
        torch::ScalarType dtype = input.scalar_type();
        if (dtype == torch::kInt8 || dtype == torch::kUInt8 || 
            dtype == torch::kInt16 || dtype == torch::kInt32 || 
            dtype == torch::kInt64 || dtype == torch::kBool) {
            input = input.to(torch::kFloat);
        }
        
        // Ensure input is contiguous
        input = input.contiguous();
        
        // Test 1: Basic matrix_rank call without tolerance
        try {
            torch::Tensor result1 = torch::linalg_matrix_rank(input);
            (void)result1;
        } catch (const std::exception&) {
            // Shape/dtype issues are expected for some inputs
        }
        
        // Test 2: matrix_rank with tolerance as tensor
        try {
            torch::Tensor tol_tensor = torch::tensor({tol}, input.options().dtype(torch::kFloat64));
            torch::Tensor result2 = torch::linalg_matrix_rank(input, tol_tensor, hermitian);
            (void)result2;
        } catch (const std::exception&) {
            // Expected for invalid configurations
        }
        
        // Test 3: For square matrices, test hermitian path more thoroughly
        int64_t last_dim = input.size(-1);
        int64_t second_last_dim = input.size(-2);
        
        if (last_dim == second_last_dim && last_dim > 0) {
            try {
                // Make input symmetric/hermitian for valid hermitian=true case
                torch::Tensor symmetric = (input + input.transpose(-2, -1)) / 2.0;
                torch::Tensor result3 = torch::linalg_matrix_rank(symmetric, c10::nullopt, true);
                (void)result3;
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
        
        // Test 4: Test with double precision
        try {
            torch::Tensor input_double = input.to(torch::kFloat64);
            torch::Tensor result4 = torch::linalg_matrix_rank(input_double);
            (void)result4;
        } catch (const std::exception&) {
            // Expected for some inputs
        }
        
        // Test 5: Test with batched input (reshape to add batch dimension)
        if (input.dim() == 2 && input.numel() >= 4) {
            try {
                int64_t m = input.size(0);
                int64_t n = input.size(1);
                if (m >= 2 && n >= 2) {
                    // Create a batched version
                    torch::Tensor batched = input.unsqueeze(0).expand({2, m, n}).contiguous();
                    torch::Tensor result5 = torch::linalg_matrix_rank(batched);
                    (void)result5;
                }
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}