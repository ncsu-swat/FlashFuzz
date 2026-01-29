#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Need at least some data to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters from fuzzer data first
        bool some = Data[offset++] & 0x1;  // reduced vs full SVD
        bool compute_uv = Data[offset++] & 0x1;  // whether to compute U and V
        uint8_t dtype_selector = Data[offset++] % 2;  // 0=float, 1=double
        
        // Create input tensor for SVD
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // SVD requires at least 2D tensor
        if (input_tensor.dim() < 2) {
            if (input_tensor.dim() == 0) {
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0);
            } else if (input_tensor.dim() == 1) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Convert to appropriate dtype - SVD works on floating point types
        if (dtype_selector == 0) {
            input_tensor = input_tensor.to(torch::kFloat32);
        } else {
            input_tensor = input_tensor.to(torch::kFloat64);
        }
        
        // Ensure tensor is not too large to avoid memory issues
        // and has reasonable dimensions for SVD
        auto sizes = input_tensor.sizes();
        int64_t m = sizes[sizes.size() - 2];
        int64_t n = sizes[sizes.size() - 1];
        
        // Skip very large matrices to avoid timeouts
        if (m > 64 || n > 64) {
            input_tensor = input_tensor.index({"...", 
                torch::indexing::Slice(0, std::min(m, (int64_t)32)),
                torch::indexing::Slice(0, std::min(n, (int64_t)32))});
        }
        
        // Handle potential NaN/Inf values that could cause SVD to fail
        input_tensor = torch::nan_to_num(input_tensor, 0.0, 1.0, -1.0);
        
        // Apply SVD operation
        // torch::svd returns tuple of (U, S, V)
        // 'some' parameter: if true, returns reduced SVD
        // 'compute_uv' parameter: if false, U and V are empty tensors
        auto svd_result = torch::svd(input_tensor, some, compute_uv);
        
        // Unpack the results
        torch::Tensor U = std::get<0>(svd_result);
        torch::Tensor S = std::get<1>(svd_result);
        torch::Tensor V = std::get<2>(svd_result);
        
        // Basic validation - singular values should be non-negative
        if (S.numel() > 0) {
            auto min_s = S.min().item<double>();
            // Singular values should be >= 0 (allowing small numerical error)
            (void)min_s;  // Use the result to prevent optimization
        }
        
        // If U and V were computed, try reconstruction
        if (compute_uv && U.numel() > 0 && V.numel() > 0) {
            try {
                // For reconstruction: A ≈ U @ diag(S) @ V^T
                // Get the dimensions
                auto k = S.size(-1);  // number of singular values
                (void)k;
                
                // Create diagonal matrix from singular values
                torch::Tensor S_diag;
                if (S.dim() == 1) {
                    S_diag = torch::diag(S);
                } else {
                    // Batched case
                    S_diag = torch::diag_embed(S);
                }
                
                // Transpose V
                auto V_t = V.transpose(-2, -1);
                
                // Slice U and V to match S dimensions for reduced SVD
                if (some) {
                    // For reduced SVD, dimensions should already match
                    auto reconstructed = torch::matmul(torch::matmul(U, S_diag), V_t);
                    
                    // Compute reconstruction error
                    auto error = torch::norm(reconstructed - input_tensor).item<double>();
                    (void)error;  // Use result
                }
            } catch (...) {
                // Reconstruction may fail for edge cases, that's okay
            }
        }
        
        // Test with different input configurations
        if (offset < Size) {
            try {
                // Test with a transposed input
                auto transposed_input = input_tensor.transpose(-2, -1).contiguous();
                auto svd_transposed = torch::svd(transposed_input, some, compute_uv);
                
                torch::Tensor S_t = std::get<1>(svd_transposed);
                if (S_t.numel() > 0) {
                    auto sum_s = S_t.sum().item<double>();
                    (void)sum_s;
                }
            } catch (...) {
                // Some edge cases may fail
            }
        }
        
        // Test with complex tensor if available
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                // Create a complex tensor from input
                auto real_part = input_tensor;
                auto imag_part = input_tensor * 0.5;
                auto complex_input = torch::complex(real_part.to(torch::kFloat32), 
                                                     imag_part.to(torch::kFloat32));
                
                auto svd_complex = torch::svd(complex_input, some, compute_uv);
                torch::Tensor S_c = std::get<1>(svd_complex);
                if (S_c.numel() > 0) {
                    // Singular values are always real
                    auto max_s = S_c.max().item<double>();
                    (void)max_s;
                }
            } catch (...) {
                // Complex SVD may have different requirements
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