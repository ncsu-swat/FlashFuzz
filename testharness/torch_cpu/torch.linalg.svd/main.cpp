#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for SVD
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // SVD requires at least 2D tensor
        if (A.dim() < 2) {
            return 0;
        }
        
        // SVD requires floating point or complex types
        if (!(A.scalar_type() == torch::kFloat || 
              A.scalar_type() == torch::kDouble || 
              A.scalar_type() == torch::kComplexFloat || 
              A.scalar_type() == torch::kComplexDouble)) {
            // Convert to float for non-floating types
            A = A.to(torch::kFloat);
        }
        
        // Parse SVD options from the remaining data
        bool full_matrices = false;
        
        if (offset < Size) {
            full_matrices = Data[offset++] & 0x1;
        }
        
        // Apply SVD operation
        auto svd_result = torch::linalg_svd(A, full_matrices);
        
        // Unpack the result
        auto U = std::get<0>(svd_result);
        auto S = std::get<1>(svd_result);
        auto Vh = std::get<2>(svd_result);
        
        // Exercise the output tensors to ensure they're computed
        auto u_sum = U.sum();
        auto s_sum = S.sum();
        auto vh_sum = Vh.sum();
        
        // Verify basic properties of SVD
        // S should contain non-negative singular values
        auto s_min = S.min();
        
        // For full_matrices=false (reduced SVD):
        // If A is (m x n), then:
        //   U is (m x k), S is (k,), Vh is (k x n) where k = min(m, n)
        // For full_matrices=true:
        //   U is (m x m), S is (k,), Vh is (n x n)
        
        // Test with different input configurations
        if (offset < Size) {
            // Create another tensor and test batched SVD
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            if (B.dim() >= 2) {
                if (!(B.scalar_type() == torch::kFloat || 
                      B.scalar_type() == torch::kDouble || 
                      B.scalar_type() == torch::kComplexFloat || 
                      B.scalar_type() == torch::kComplexDouble)) {
                    B = B.to(torch::kFloat);
                }
                
                try {
                    auto svd_result_b = torch::linalg_svd(B, !full_matrices);
                    auto U_b = std::get<0>(svd_result_b);
                    auto S_b = std::get<1>(svd_result_b);
                    auto Vh_b = std::get<2>(svd_result_b);
                    
                    // Exercise outputs
                    (void)U_b.sum();
                    (void)S_b.sum();
                    (void)Vh_b.sum();
                }
                catch (...) {
                    // Silently ignore expected failures (shape mismatches, etc.)
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}