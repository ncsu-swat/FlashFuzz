#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        
        // Test various torch linalg operations based on the selector
        switch (op_selector % 20) {
            case 0: {
                // torch.norm
                torch::Tensor result = torch::norm(A);
                break;
            }
            case 1: {
                // torch.matrix_norm with different ord values
                if (A.dim() >= 2) {
                    int ord = param % 3;
                    torch::Tensor result = torch::norm(A, ord);
                }
                break;
            }
            case 2: {
                // torch.det (determinant)
                if (A.dim() >= 2) {
                    torch::Tensor result = torch::det(A);
                }
                break;
            }
            case 3: {
                // torch.slogdet (sign of determinant and log-determinant)
                if (A.dim() >= 2) {
                    auto result = torch::slogdet(A);
                }
                break;
            }
            case 4: {
                // torch.matrix_rank
                if (A.dim() >= 2) {
                    torch::Tensor result = torch::matrix_rank(A);
                }
                break;
            }
            case 5: {
                // torch.svd (singular value decomposition)
                if (A.dim() >= 2) {
                    bool compute_uv = param % 2 == 0;
                    auto result = torch::svd(A, compute_uv);
                }
                break;
            }
            case 6: {
                // torch.eig (eigenvalues and eigenvectors)
                if (A.dim() >= 2) {
                    bool eigenvectors = param % 2 == 0;
                    auto result = torch::eig(A, eigenvectors);
                }
                break;
            }
            case 7: {
                // torch.symeig (symmetric eigenvalues)
                if (A.dim() >= 2) {
                    bool eigenvectors = param % 2 == 0;
                    auto result = torch::symeig(A, eigenvectors);
                }
                break;
            }
            case 8: {
                // torch.inverse (matrix inverse)
                if (A.dim() >= 2) {
                    torch::Tensor result = torch::inverse(A);
                }
                break;
            }
            case 9: {
                // torch.pinverse (pseudo-inverse)
                if (A.dim() >= 2) {
                    torch::Tensor result = torch::pinverse(A);
                }
                break;
            }
            case 10: {
                // torch.matrix_power
                if (A.dim() >= 2) {
                    int n = param % 5 - 2; // Powers between -2 and 2
                    torch::Tensor result = torch::matrix_power(A, n);
                }
                break;
            }
            case 11: {
                // torch.solve
                if (A.dim() >= 2 && B.defined() && B.dim() >= 1) {
                    try {
                        auto result = torch::solve(B, A);
                    } catch (...) {
                        // Solving might fail for non-invertible matrices
                    }
                }
                break;
            }
            case 12: {
                // torch.cholesky
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::cholesky(A);
                    } catch (...) {
                        // Cholesky might fail for non-positive-definite matrices
                    }
                }
                break;
            }
            case 13: {
                // torch.qr
                if (A.dim() >= 2) {
                    auto result = torch::qr(A);
                }
                break;
            }
            case 14: {
                // torch.lu
                if (A.dim() >= 2) {
                    auto result = torch::lu(A);
                }
                break;
            }
            case 15: {
                // torch.lu_unpack
                if (A.dim() >= 2) {
                    try {
                        auto lu_result = torch::lu(A);
                        auto result = torch::lu_unpack(std::get<0>(lu_result), std::get<1>(lu_result));
                    } catch (...) {
                        // LU decomposition might fail
                    }
                }
                break;
            }
            case 16: {
                // torch.cross
                if (A.dim() >= 1 && B.defined() && B.dim() >= 1) {
                    try {
                        torch::Tensor result = torch::cross(A, B);
                    } catch (...) {
                        // Cross product has specific dimension requirements
                    }
                }
                break;
            }
            case 17: {
                // torch.norm with vector norm
                torch::Tensor result = torch::norm(A);
                break;
            }
            case 18: {
                // torch.cond (condition number)
                if (A.dim() >= 2) {
                    try {
                        torch::Tensor result = torch::cond(A);
                    } catch (...) {
                        // Condition number might fail for singular matrices
                    }
                }
                break;
            }
            case 19: {
                // torch.chain_matmul (multiple matrix multiplication)
                if (A.dim() >= 2 && B.defined() && B.dim() >= 2) {
                    std::vector<torch::Tensor> tensors = {A, B};
                    try {
                        torch::Tensor result = torch::chain_matmul(tensors);
                    } catch (...) {
                        // Matrix multiplication has specific dimension requirements
                    }
                }
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
