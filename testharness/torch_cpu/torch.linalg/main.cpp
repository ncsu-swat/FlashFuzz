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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if there's enough data left
        torch::Tensor B;
        if (offset + 2 < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Get a selector byte for choosing which linalg operation to test
        uint8_t op_selector = 0;
        if (offset < Size) {
            op_selector = Data[offset++];
        }
        
        // Get a parameter byte for additional options
        uint8_t param = 0;
        if (offset < Size) {
            param = Data[offset++];
        }
        
        // Ensure tensors are float for most linalg operations
        if (A.scalar_type() != torch::kFloat && A.scalar_type() != torch::kDouble) {
            A = A.to(torch::kFloat);
        }
        if (B.defined() && B.scalar_type() != torch::kFloat && B.scalar_type() != torch::kDouble) {
            B = B.to(torch::kFloat);
        }
        
        // Test various torch linalg operations based on the selector
        // Note: In C++ frontend, linalg functions use torch::linalg_* naming convention
        switch (op_selector % 20) {
            case 0: {
                // torch.linalg.norm - vector norm (use torch::norm as fallback)
                torch::Tensor result = torch::norm(A);
                break;
            }
            case 1: {
                // torch.linalg.det (determinant) - available as torch::linalg_det
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_det(A);
                    } catch (...) {
                        // May fail for non-square matrices
                    }
                }
                break;
            }
            case 2: {
                // torch.linalg.slogdet - available as torch::linalg_slogdet
                if (A.dim() >= 2) {
                    try {
                        auto result = torch::linalg_slogdet(A);
                    } catch (...) {
                        // May fail for non-square matrices
                    }
                }
                break;
            }
            case 3: {
                // torch.linalg.matrix_rank - available as torch::linalg_matrix_rank
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_matrix_rank(A);
                    } catch (...) {
                        // May fail for certain matrix shapes
                    }
                }
                break;
            }
            case 4: {
                // torch.linalg.svd - available as torch::linalg_svd
                if (A.dim() >= 2) {
                    try {
                        bool full_matrices = param % 2 == 0;
                        auto result = torch::linalg_svd(A, full_matrices);
                    } catch (...) {
                        // SVD may fail for certain matrices
                    }
                }
                break;
            }
            case 5: {
                // torch.linalg.eig - available as torch::linalg_eig
                if (A.dim() >= 2) {
                    try {
                        auto result = torch::linalg_eig(A);
                    } catch (...) {
                        // May fail for non-square matrices
                    }
                }
                break;
            }
            case 6: {
                // torch.linalg.eigh - available as torch::linalg_eigh
                if (A.dim() >= 2) {
                    try {
                        std::string uplo = (param % 2 == 0) ? "U" : "L";
                        auto result = torch::linalg_eigh(A, uplo);
                    } catch (...) {
                        // May fail for non-symmetric or non-square matrices
                    }
                }
                break;
            }
            case 7: {
                // torch.linalg.inv - available as torch::linalg_inv
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_inv(A);
                    } catch (...) {
                        // May fail for singular matrices
                    }
                }
                break;
            }
            case 8: {
                // torch.linalg.pinv - available as torch::linalg_pinv
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_pinv(A);
                    } catch (...) {
                        // May fail for certain matrices
                    }
                }
                break;
            }
            case 9: {
                // torch.linalg.matrix_power - available as torch::linalg_matrix_power
                if (A.dim() >= 2) {
                    try {
                        int64_t n = (param % 5) - 2; // Powers between -2 and 2
                        torch::Tensor result = torch::linalg_matrix_power(A, n);
                    } catch (...) {
                        // May fail for non-square or singular matrices with negative powers
                    }
                }
                break;
            }
            case 10: {
                // torch.linalg.solve - available as torch::linalg_solve
                if (A.dim() >= 2 && B.defined() && B.dim() >= 1) {
                    try {
                        auto result = torch::linalg_solve(A, B);
                    } catch (...) {
                        // Solving might fail for non-invertible matrices or shape mismatch
                    }
                }
                break;
            }
            case 11: {
                // torch.linalg.cholesky - available as torch::linalg_cholesky
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_cholesky(A);
                    } catch (...) {
                        // Cholesky fails for non-positive-definite matrices
                    }
                }
                break;
            }
            case 12: {
                // torch.linalg.qr - available as torch::linalg_qr
                if (A.dim() >= 2) {
                    try {
                        std::string mode = (param % 2 == 0) ? "reduced" : "complete";
                        auto result = torch::linalg_qr(A, mode);
                    } catch (...) {
                        // QR may fail for certain matrices
                    }
                }
                break;
            }
            case 13: {
                // torch.linalg.lu - available as torch::linalg_lu
                if (A.dim() >= 2) {
                    try {
                        auto result = torch::linalg_lu(A);
                    } catch (...) {
                        // LU may fail for certain matrices
                    }
                }
                break;
            }
            case 14: {
                // torch.linalg.lu_factor - available as torch::linalg_lu_factor
                if (A.dim() >= 2) {
                    try {
                        auto result = torch::linalg_lu_factor(A);
                    } catch (...) {
                        // LU factorization may fail
                    }
                }
                break;
            }
            case 15: {
                // torch.linalg.cross - available as torch::linalg_cross
                if (A.dim() >= 1 && B.defined() && B.dim() >= 1) {
                    try {
                        torch::Tensor result = torch::linalg_cross(A, B);
                    } catch (...) {
                        // Cross product has specific dimension requirements (size 3)
                    }
                }
                break;
            }
            case 16: {
                // torch.linalg.vector_norm - available as torch::linalg_vector_norm
                try {
                    double ord_vals[] = {1.0, 2.0, std::numeric_limits<double>::infinity()};
                    double ord = ord_vals[param % 3];
                    torch::Tensor result = torch::linalg_vector_norm(A, ord);
                } catch (...) {
                    // May fail for certain inputs
                }
                break;
            }
            case 17: {
                // torch.linalg.cond - available as torch::linalg_cond
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_cond(A);
                    } catch (...) {
                        // Condition number might fail for singular matrices
                    }
                }
                break;
            }
            case 18: {
                // torch.linalg.eigvals - available as torch::linalg_eigvals
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_eigvals(A);
                    } catch (...) {
                        // May fail for non-square matrices
                    }
                }
                break;
            }
            case 19: {
                // torch.linalg.matrix_norm - available as torch::linalg_matrix_norm
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::linalg_matrix_norm(A);
                    } catch (...) {
                        // May fail for certain matrix shapes
                    }
                }
                break;
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