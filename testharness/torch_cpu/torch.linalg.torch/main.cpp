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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try different torch operations based on remaining data
        if (offset < Size) {
            uint8_t op_selector = Data[offset++] % 5;
            
            switch (op_selector) {
                case 0: {
                    // torch.norm
                    torch::Tensor result = torch::norm(input);
                    break;
                }
                case 1: {
                    // torch.det (requires square matrix)
                    if (input.dim() >= 2 && input.size(0) == input.size(1)) {
                        torch::Tensor result = torch::det(input);
                    }
                    break;
                }
                case 2: {
                    // torch.inverse (requires square matrix)
                    if (input.dim() >= 2 && input.size(0) == input.size(1)) {
                        torch::Tensor result = torch::inverse(input);
                    }
                    break;
                }
                case 3: {
                    // torch.svd
                    if (input.dim() >= 2) {
                        auto result = torch::svd(input);
                        torch::Tensor U = std::get<0>(result);
                        torch::Tensor S = std::get<1>(result);
                        torch::Tensor V = std::get<2>(result);
                    }
                    break;
                }
                case 4: {
                    // torch.qr
                    if (input.dim() >= 2) {
                        auto result = torch::qr(input);
                        torch::Tensor Q = std::get<0>(result);
                        torch::Tensor R = std::get<1>(result);
                    }
                    break;
                }
            }
        }
        
        // Try another operation if we have more data
        if (offset + 1 < Size) {
            uint8_t op_selector = Data[offset++] % 5;
            
            switch (op_selector) {
                case 0: {
                    // torch.matrix_rank
                    if (input.dim() >= 2) {
                        torch::Tensor result = torch::matrix_rank(input);
                    }
                    break;
                }
                case 1: {
                    // torch.cholesky (requires positive-definite matrix)
                    if (input.dim() >= 2 && input.size(0) == input.size(1)) {
                        // Try to make it positive definite
                        torch::Tensor A = torch::matmul(input, input.transpose(-2, -1));
                        // Add small positive value to diagonal for numerical stability
                        A = A + torch::eye(A.size(0), A.options()) * 1e-3;
                        torch::Tensor result = torch::cholesky(A);
                    }
                    break;
                }
                case 2: {
                    // torch.symeig (requires square matrix)
                    if (input.dim() >= 2 && input.size(0) == input.size(1)) {
                        // Make symmetric for symeig
                        torch::Tensor A = input + input.transpose(-2, -1);
                        auto result = torch::symeig(A, true);
                        torch::Tensor eigenvalues = std::get<0>(result);
                        torch::Tensor eigenvectors = std::get<1>(result);
                    }
                    break;
                }
                case 3: {
                    // torch.solve
                    if (input.dim() >= 2 && input.size(0) == input.size(1)) {
                        // Create a right-hand side vector/matrix
                        std::vector<int64_t> b_shape = input.sizes().vec();
                        b_shape.back() = 1; // Make it a column vector
                        torch::Tensor b = torch::ones(b_shape, input.options());
                        auto result = torch::solve(b, input);
                        torch::Tensor solution = std::get<0>(result);
                    }
                    break;
                }
                case 4: {
                    // torch.lu
                    if (input.dim() >= 2) {
                        auto result = torch::lu(input);
                        torch::Tensor LU = std::get<0>(result);
                        torch::Tensor pivots = std::get<1>(result);
                    }
                    break;
                }
            }
        }
        
        // Try a third operation if we have even more data
        if (offset + 1 < Size) {
            uint8_t op_selector = Data[offset++] % 3;
            
            switch (op_selector) {
                case 0: {
                    // torch.pinverse (pseudoinverse)
                    torch::Tensor result = torch::pinverse(input);
                    break;
                }
                case 1: {
                    // torch.matrix_power (requires square matrix)
                    if (input.dim() >= 2 && input.size(0) == input.size(1)) {
                        // Use a small power to avoid numerical issues
                        int64_t n = 2;
                        if (offset < Size) {
                            n = static_cast<int64_t>(Data[offset++]) % 5;
                        }
                        torch::Tensor result = torch::matrix_power(input, n);
                    }
                    break;
                }
                case 2: {
                    // torch.norm with ord parameter
                    if (input.dim() >= 1) {
                        double ord = 2.0; // Default L2 norm
                        if (offset < Size) {
                            ord = static_cast<double>(Data[offset++] % 3);
                        }
                        torch::Tensor result = torch::norm(input, ord);
                    }
                    break;
                }
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
