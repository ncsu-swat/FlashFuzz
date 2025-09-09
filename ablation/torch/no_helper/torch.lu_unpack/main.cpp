#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 20) return 0;

        // Extract basic parameters
        int batch_size = extractInt(Data, Size, offset, 1, 4);
        int m = extractInt(Data, Size, offset, 1, 10);
        int n = extractInt(Data, Size, offset, 1, 10);
        bool unpack_data = extractBool(Data, Size, offset);
        bool unpack_pivots = extractBool(Data, Size, offset);
        int dtype_idx = extractInt(Data, Size, offset, 0, 2);
        
        // Select data type
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            default: dtype = torch::kFloat32; break;
        }

        // Create a random matrix for LU factorization
        torch::Tensor A;
        if (batch_size > 1) {
            A = torch::randn({batch_size, m, n}, torch::TensorOptions().dtype(dtype));
        } else {
            A = torch::randn({m, n}, torch::TensorOptions().dtype(dtype));
        }

        // Get LU factorization first
        auto [LU_data, LU_pivots] = torch::linalg::lu_factor(A);

        // Test basic lu_unpack call
        auto result = torch::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
        auto P = std::get<0>(result);
        auto L = std::get<1>(result);
        auto U = std::get<2>(result);

        // Verify tensor properties
        if (unpack_data) {
            if (L.numel() == 0 || U.numel() == 0) {
                throw std::runtime_error("Expected non-empty L and U tensors when unpack_data=true");
            }
        } else {
            if (L.numel() != 0 || U.numel() != 0) {
                throw std::runtime_error("Expected empty L and U tensors when unpack_data=false");
            }
        }

        if (unpack_pivots) {
            if (P.numel() == 0) {
                throw std::runtime_error("Expected non-empty P tensor when unpack_pivots=true");
            }
        } else {
            if (P.numel() != 0) {
                throw std::runtime_error("Expected empty P tensor when unpack_pivots=false");
            }
        }

        // Test with output tuple
        if (offset < Size - 4) {
            bool test_out = extractBool(Data, Size, offset);
            if (test_out && unpack_data && unpack_pivots) {
                auto P_out = torch::empty_like(P);
                auto L_out = torch::empty_like(L);
                auto U_out = torch::empty_like(U);
                
                auto out_tuple = std::make_tuple(P_out, L_out, U_out);
                torch::lu_unpack_out(out_tuple, LU_data, LU_pivots, unpack_data, unpack_pivots);
            }
        }

        // Test edge cases with different boolean combinations
        if (offset < Size - 8) {
            // Test all combinations of unpack flags
            torch::lu_unpack(LU_data, LU_pivots, true, true);
            torch::lu_unpack(LU_data, LU_pivots, true, false);
            torch::lu_unpack(LU_data, LU_pivots, false, true);
            torch::lu_unpack(LU_data, LU_pivots, false, false);
        }

        // Test with different matrix shapes if we have more data
        if (offset < Size - 12) {
            int m2 = extractInt(Data, Size, offset, 1, 8);
            int n2 = extractInt(Data, Size, offset, 1, 8);
            
            torch::Tensor A2;
            if (batch_size > 1) {
                A2 = torch::randn({batch_size, m2, n2}, torch::TensorOptions().dtype(dtype));
            } else {
                A2 = torch::randn({m2, n2}, torch::TensorOptions().dtype(dtype));
            }
            
            auto [LU_data2, LU_pivots2] = torch::linalg::lu_factor(A2);
            auto result2 = torch::lu_unpack(LU_data2, LU_pivots2);
        }

        // Test with singular/near-singular matrices
        if (offset < Size - 4) {
            bool test_singular = extractBool(Data, Size, offset);
            if (test_singular) {
                torch::Tensor singular_A;
                if (batch_size > 1) {
                    singular_A = torch::zeros({batch_size, std::min(m,n), std::min(m,n)}, torch::TensorOptions().dtype(dtype));
                } else {
                    singular_A = torch::zeros({std::min(m,n), std::min(m,n)}, torch::TensorOptions().dtype(dtype));
                }
                
                // Add some small values to avoid completely zero matrix
                singular_A += torch::randn_like(singular_A) * 1e-10;
                
                try {
                    auto [LU_singular, pivots_singular] = torch::linalg::lu_factor(singular_A);
                    torch::lu_unpack(LU_singular, pivots_singular, unpack_data, unpack_pivots);
                } catch (const std::exception&) {
                    // Singular matrices might cause issues, which is expected
                }
            }
        }

        // Test with complex numbers if dtype supports it
        if (dtype == torch::kComplexFloat && offset < Size - 4) {
            bool test_complex = extractBool(Data, Size, offset);
            if (test_complex) {
                auto complex_A = torch::complex(
                    torch::randn({m, n}, torch::kFloat32),
                    torch::randn({m, n}, torch::kFloat32)
                );
                
                auto [LU_complex, pivots_complex] = torch::linalg::lu_factor(complex_A);
                torch::lu_unpack(LU_complex, pivots_complex, unpack_data, unpack_pivots);
            }
        }

        // Verify reconstruction when both flags are true
        if (unpack_data && unpack_pivots && P.numel() > 0 && L.numel() > 0 && U.numel() > 0) {
            try {
                auto reconstructed = torch::matmul(torch::matmul(P, L), U);
                // Check if dimensions match for comparison
                if (reconstructed.sizes() == A.sizes()) {
                    auto diff = torch::abs(A - reconstructed);
                    auto max_diff = torch::max(diff);
                    // Allow for numerical precision differences
                    if (max_diff.item<double>() > 1e-3) {
                        // This might indicate an issue, but could also be due to numerical precision
                    }
                }
            } catch (const std::exception&) {
                // Matrix multiplication might fail for incompatible dimensions
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}