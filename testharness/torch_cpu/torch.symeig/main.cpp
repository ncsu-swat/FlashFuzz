#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        // Need minimum data for meaningful test
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse parameters from input data
        bool upper = Data[offset++] & 0x1;
        uint8_t size_hint = Data[offset++];
        
        // Determine matrix size (1 to 16 for reasonable computation time)
        int64_t n = (size_hint % 15) + 1;
        
        // Create a tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // Ensure we have a floating point type for eigenvalue computation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Handle complex types - convert to real
        if (input.is_complex()) {
            input = torch::real(input);
        }
        
        // Reshape to square matrix
        int64_t total_elements = input.numel();
        if (total_elements == 0) {
            // Create a small default matrix
            input = torch::randn({n, n}, torch::kFloat);
        } else if (total_elements < n * n) {
            // Repeat elements to fill the matrix
            int64_t actual_n = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
            if (actual_n < 1) actual_n = 1;
            n = actual_n;
            input = input.flatten().narrow(0, 0, std::min(total_elements, n * n));
            if (input.numel() < n * n) {
                // Pad with zeros
                torch::Tensor padded = torch::zeros({n * n}, input.options());
                padded.narrow(0, 0, input.numel()).copy_(input);
                input = padded;
            }
            input = input.reshape({n, n});
        } else {
            // Use first n*n elements
            input = input.flatten().narrow(0, 0, n * n).reshape({n, n});
        }
        
        // Make the matrix symmetric: A = (M + M^T) / 2
        torch::Tensor symmetric_input = (input + input.transpose(0, 1)) * 0.5;
        
        // Ensure no NaN or Inf values
        if (torch::any(torch::isnan(symmetric_input)).item<bool>() ||
            torch::any(torch::isinf(symmetric_input)).item<bool>()) {
            symmetric_input = torch::randn({n, n}, torch::kFloat);
            symmetric_input = (symmetric_input + symmetric_input.transpose(0, 1)) * 0.5;
        }
        
        // Call torch::linalg_eigh (replacement for deprecated torch::symeig)
        // UPLO parameter: "U" for upper triangular, "L" for lower triangular
        std::tuple<torch::Tensor, torch::Tensor> result;
        
        try {
            result = torch::linalg_eigh(symmetric_input, upper ? "U" : "L");
        } catch (const std::exception &) {
            // Expected failure for some edge cases (singular matrices, etc.)
            return 0;
        }
        
        torch::Tensor eigenvalues = std::get<0>(result);
        torch::Tensor eigenvectors = std::get<1>(result);
        
        // Basic validation - eigenvalues should be real for symmetric matrices
        // and eigenvectors should be orthogonal
        if (eigenvalues.numel() > 0 && eigenvectors.numel() > 0) {
            // Verify reconstruction: A = V * diag(eigenvalues) * V^T
            torch::Tensor reconstructed = torch::matmul(
                torch::matmul(eigenvectors, torch::diag(eigenvalues)),
                eigenvectors.transpose(0, 1)
            );
            
            // Compute difference (for coverage, not validation)
            torch::Tensor diff = symmetric_input - reconstructed;
            volatile float max_diff = torch::max(torch::abs(diff)).item<float>();
            (void)max_diff;  // Prevent optimization
            
            // Check orthogonality of eigenvectors
            torch::Tensor identity_check = torch::matmul(
                eigenvectors.transpose(0, 1), 
                eigenvectors
            );
            volatile float trace_val = torch::trace(identity_check).item<float>();
            (void)trace_val;  // Prevent optimization
        }
        
        // Also test batched input for better coverage
        if (n <= 8 && Size > 10) {
            torch::Tensor batched = symmetric_input.unsqueeze(0).expand({2, n, n}).clone();
            
            try {
                auto batched_result = torch::linalg_eigh(batched, upper ? "U" : "L");
                torch::Tensor batched_eigenvalues = std::get<0>(batched_result);
                torch::Tensor batched_eigenvectors = std::get<1>(batched_result);
                
                // Access results to ensure computation
                volatile float first_eigenvalue = batched_eigenvalues[0][0].item<float>();
                (void)first_eigenvalue;
            } catch (const std::exception &) {
                // Expected for some edge cases
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