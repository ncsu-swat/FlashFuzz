#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Determine matrix size from input data
        uint8_t dim_byte = Data[offset++];
        int64_t n = (dim_byte % 8) + 1;  // Matrix size 1x1 to 8x8
        
        // Determine batch dimensions
        uint8_t batch_byte = Data[offset++];
        int64_t batch = (batch_byte % 4) + 1;  // Batch size 1 to 4
        
        // Create input tensor with appropriate shape for inv_ex
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape to a valid square matrix (batch x n x n)
        int64_t total_elements = A.numel();
        int64_t needed_elements = batch * n * n;
        
        if (total_elements < needed_elements) {
            // Pad with random values if we don't have enough elements
            A = torch::cat({A.flatten(), torch::randn({needed_elements - total_elements}, A.options())});
        }
        
        // Take only needed elements and reshape to square matrix
        A = A.flatten().slice(0, 0, needed_elements).reshape({batch, n, n});
        
        // Convert to float type suitable for linalg operations
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat);
        }
        
        // Test torch::linalg_inv_ex - returns (inverse, info)
        // check_errors=false allows us to get info tensor instead of throwing
        auto result = torch::linalg_inv_ex(A, /*check_errors=*/false);
        torch::Tensor inverse = std::get<0>(result);
        torch::Tensor info = std::get<1>(result);
        
        // Use results to prevent optimization
        auto inv_sum = inverse.sum();
        auto info_sum = info.sum();
        if (inv_sum.item<float>() == -12345.6789f) {
            std::cerr << "Unreachable" << std::endl;
        }
        
        // Test with check_errors=true (may throw for singular matrices)
        try {
            auto result_checked = torch::linalg_inv_ex(A, /*check_errors=*/true);
            torch::Tensor inverse_checked = std::get<0>(result_checked);
            auto sum_checked = inverse_checked.sum();
            if (sum_checked.item<float>() == -12345.6789f) {
                std::cerr << "Unreachable" << std::endl;
            }
        } catch (const std::exception &) {
            // Expected for singular matrices - silently ignore
        }
        
        // Test with double precision
        if (offset < Size) {
            auto A_double = A.to(torch::kDouble);
            auto result_double = torch::linalg_inv_ex(A_double, /*check_errors=*/false);
            torch::Tensor inverse_double = std::get<0>(result_double);
            torch::Tensor info_double = std::get<1>(result_double);
            
            auto sum_double = inverse_double.sum();
            if (sum_double.item<double>() == -12345.6789) {
                std::cerr << "Unreachable" << std::endl;
            }
        }
        
        // Test with well-conditioned matrix (add scaled identity)
        if (offset + 1 < Size) {
            uint8_t scale_byte = Data[offset++];
            float scale = (scale_byte / 255.0f) * 10.0f + 0.1f;
            
            auto identity = torch::eye(n, A.options());
            identity = identity.unsqueeze(0).expand({batch, n, n});
            auto A_conditioned = A + scale * identity;
            
            auto result_cond = torch::linalg_inv_ex(A_conditioned, /*check_errors=*/false);
            torch::Tensor inverse_cond = std::get<0>(result_cond);
            auto sum_cond = inverse_cond.sum();
            if (sum_cond.item<float>() == -12345.6789f) {
                std::cerr << "Unreachable" << std::endl;
            }
        }
        
        // Test with complex tensors if supported
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            try {
                auto A_complex = torch::complex(A, torch::zeros_like(A));
                auto result_complex = torch::linalg_inv_ex(A_complex, /*check_errors=*/false);
                torch::Tensor inverse_complex = std::get<0>(result_complex);
                auto sum_complex = inverse_complex.sum();
                // Just accessing to prevent optimization
                (void)sum_complex;
            } catch (const std::exception &) {
                // Complex may not be supported in all configurations
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