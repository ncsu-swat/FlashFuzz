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
        
        // Parse parameters first
        bool hermitian = Data[offset++] & 0x1;
        bool check_errors = Data[offset++] & 0x1;
        uint8_t size_hint = Data[offset++];
        
        // Determine matrix size (2 to 10)
        int64_t n = 2 + (size_hint % 9);
        
        // Create a tensor from fuzzer data
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if needed (LDL requires floating point)
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat32);
        }
        
        // Reshape to square matrix
        if (A.numel() == 0) {
            A = torch::eye(n, A.options().dtype(torch::kFloat32));
        } else if (A.dim() < 2) {
            // Reshape 1D tensor to square matrix
            int64_t total = A.numel();
            int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(total)));
            if (side < 2) side = 2;
            // Pad or truncate to make it square
            A = A.flatten();
            if (A.numel() < side * side) {
                A = torch::cat({A, torch::zeros({side * side - A.numel()}, A.options())});
            }
            A = A.slice(0, 0, side * side).reshape({side, side});
        } else {
            // For 2D+ tensors, take a square slice
            auto sizes = A.sizes().vec();
            int64_t rows = sizes[sizes.size() - 2];
            int64_t cols = sizes[sizes.size() - 1];
            int64_t min_dim = std::min(rows, cols);
            if (min_dim < 2) min_dim = 2;
            
            // Slice to make square
            A = A.slice(-2, 0, min_dim).slice(-1, 0, min_dim);
        }
        
        // Make the matrix symmetric (required for LDL factorization)
        // A_sym = (A + A^T) / 2
        A = (A + A.transpose(-2, -1)) / 2.0;
        
        // Add positive value to diagonal for numerical stability
        // This helps ensure the matrix is more likely to be factorizable
        int64_t mat_size = A.size(-1);
        torch::Tensor eye_mat = torch::eye(mat_size, A.options());
        if (A.dim() > 2) {
            // Broadcast eye matrix to batch dimensions
            std::vector<int64_t> broadcast_shape(A.dim(), 1);
            broadcast_shape[A.dim() - 2] = mat_size;
            broadcast_shape[A.dim() - 1] = mat_size;
            eye_mat = eye_mat.reshape(broadcast_shape).expand(A.sizes());
        }
        A = A + eye_mat * 1.0;
        
        // Ensure contiguous
        A = A.contiguous();
        
        // Call torch::linalg_ldl_factor_ex (C++ API uses underscore naming)
        try {
            auto result = torch::linalg_ldl_factor_ex(A, hermitian, check_errors);
            
            // Unpack the result (LD, pivots, info)
            auto LD = std::get<0>(result);
            auto pivots = std::get<1>(result);
            auto info = std::get<2>(result);
            
            // Use the outputs to prevent optimization
            if (LD.numel() > 0) {
                auto sum = LD.sum();
                (void)sum;
            }
            if (pivots.numel() > 0) {
                auto p_sum = pivots.sum();
                (void)p_sum;
            }
            if (info.numel() > 0) {
                auto i_sum = info.sum();
                (void)i_sum;
            }
        } catch (const c10::Error& e) {
            // Expected errors for invalid matrices
        } catch (const std::runtime_error& e) {
            // Expected runtime errors
        }
        
        // Test with opposite hermitian flag
        try {
            auto result2 = torch::linalg_ldl_factor_ex(A, !hermitian, check_errors);
            auto LD2 = std::get<0>(result2);
            if (LD2.numel() > 0) {
                auto sum = LD2.sum();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // Expected errors
        } catch (const std::runtime_error& e) {
            // Expected runtime errors
        }
        
        // Test with complex tensor if we have enough data
        if (offset + 10 < Size) {
            try {
                torch::Tensor A_complex = torch::complex(A, A * 0.1);
                A_complex = (A_complex + A_complex.transpose(-2, -1).conj()) / 2.0;
                
                auto result3 = torch::linalg_ldl_factor_ex(A_complex, true, check_errors);
                auto LD3 = std::get<0>(result3);
                if (LD3.numel() > 0) {
                    auto sum = LD3.abs().sum();
                    (void)sum;
                }
            } catch (const c10::Error& e) {
                // Expected errors
            } catch (const std::runtime_error& e) {
                // Expected runtime errors
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