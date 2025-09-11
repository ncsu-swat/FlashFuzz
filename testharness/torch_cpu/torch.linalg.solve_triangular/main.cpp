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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create coefficient matrix A (must be triangular)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create right-hand side matrix B
        if (offset < Size) {
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Extract parameters for solve_triangular from the input data
            bool upper = true;
            bool transpose = false;
            bool unitriangular = false;
            
            if (offset < Size) {
                upper = Data[offset++] & 0x1;
            }
            
            if (offset < Size) {
                transpose = Data[offset++] & 0x1;
            }
            
            if (offset < Size) {
                unitriangular = Data[offset++] & 0x1;
            }
            
            // Make sure A has at least 2 dimensions to be a valid matrix
            if (A.dim() < 2) {
                if (A.dim() == 0) {
                    A = A.unsqueeze(0).unsqueeze(0);
                } else {
                    A = A.unsqueeze(0);
                }
            }
            
            // Make A square in the last two dimensions
            if (A.size(-1) != A.size(-2)) {
                int64_t min_dim = std::min(A.size(-1), A.size(-2));
                if (min_dim > 0) {
                    A = A.slice(-1, 0, min_dim).slice(-2, 0, min_dim);
                } else {
                    // If either dimension is 0, reshape to a 1x1 matrix
                    std::vector<int64_t> new_shape(A.dim(), 1);
                    A = A.reshape(new_shape);
                }
            }
            
            // Make sure B has compatible dimensions with A
            if (B.dim() < 1) {
                B = B.unsqueeze(0);
            }
            
            // Ensure B's last dimension matches A's last dimension
            if (B.dim() >= 1 && A.dim() >= 2) {
                if (B.size(-1) != A.size(-1)) {
                    if (A.size(-1) > 0) {
                        if (B.size(-1) > 0) {
                            B = B.slice(-1, 0, std::min(B.size(-1), A.size(-1)));
                        } else {
                            // If B's last dimension is 0, reshape it
                            std::vector<int64_t> new_shape = B.sizes().vec();
                            new_shape.back() = A.size(-1);
                            B = torch::zeros(new_shape, B.options());
                        }
                    } else {
                        // If A's last dimension is 0, reshape B
                        std::vector<int64_t> new_shape = B.sizes().vec();
                        new_shape.back() = 0;
                        B = torch::zeros(new_shape, B.options());
                    }
                }
            }
            
            // Make A triangular by zeroing out appropriate elements
            if (A.numel() > 0) {
                auto A_clone = A.clone();
                for (int64_t i = 0; i < A.size(-2); i++) {
                    for (int64_t j = 0; j < A.size(-1); j++) {
                        if ((upper && i > j) || (!upper && i < j)) {
                            // Zero out elements below/above diagonal based on 'upper'
                            if (A.dim() == 2) {
                                A_clone.index_put_({i, j}, 0);
                            } else if (A.dim() == 3) {
                                for (int64_t k = 0; k < A.size(0); k++) {
                                    A_clone.index_put_({k, i, j}, 0);
                                }
                            } else if (A.dim() == 4) {
                                for (int64_t k = 0; k < A.size(0); k++) {
                                    for (int64_t l = 0; l < A.size(1); l++) {
                                        A_clone.index_put_({k, l, i, j}, 0);
                                    }
                                }
                            }
                        }
                    }
                }
                A = A_clone;
            }
            
            // Try to solve the triangular system
            try {
                torch::Tensor X = torch::triangular_solve(B, A, upper, transpose, unitriangular).solution;
                
                // Verify the solution if possible
                if (X.numel() > 0 && A.numel() > 0 && !X.isnan().any().item<bool>() && !X.isinf().any().item<bool>()) {
                    // Compute A * X or A^T * X based on transpose flag
                    torch::Tensor A_to_use = transpose ? A.transpose(-2, -1) : A;
                    
                    // Compute the residual: A * X - B
                    torch::Tensor residual = torch::matmul(A_to_use, X) - B;
                    
                    // Check if the residual is small (solution is accurate)
                    double residual_norm = residual.norm().item<double>();
                    double b_norm = B.norm().item<double>();
                    
                    // Relative error check (avoid division by zero)
                    if (b_norm > 1e-10) {
                        double rel_error = residual_norm / b_norm;
                        if (rel_error > 1.0) {
                            // Large error might indicate numerical issues
                        }
                    }
                }
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected for invalid inputs
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
