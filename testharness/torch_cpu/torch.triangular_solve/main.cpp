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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor b
        torch::Tensor b = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor A (triangular matrix)
        torch::Tensor A;
        if (offset < Size) {
            A = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple triangular matrix
            A = torch::ones({2, 2}, b.options());
            A.triu_();
        }
        
        // Get boolean options from remaining data
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
        
        // Try to make A at least 2D if it's not already
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else if (A.dim() == 1) {
                A = A.unsqueeze(0);
            }
        }
        
        // Make sure A is square in the last two dimensions
        if (A.dim() >= 2) {
            auto last_dim = A.size(-1);
            auto second_last_dim = A.size(-2);
            
            if (last_dim != second_last_dim && last_dim > 0 && second_last_dim > 0) {
                // Resize to make square
                auto min_dim = std::min(last_dim, second_last_dim);
                std::vector<int64_t> new_sizes(A.sizes().begin(), A.sizes().end());
                new_sizes[new_sizes.size() - 1] = min_dim;
                new_sizes[new_sizes.size() - 2] = min_dim;
                A = A.index({"...", torch::indexing::Slice(0, min_dim), torch::indexing::Slice(0, min_dim)});
            }
            
            // Make sure A is triangular
            if (upper) {
                A = A.triu();
            } else {
                A = A.tril();
            }
        }
        
        // Make sure b has compatible dimensions with A
        if (b.dim() > 0 && A.dim() >= 2) {
            auto A_rows = A.size(-2);
            
            if (b.dim() == 1) {
                // For 1D b, make sure its length matches A's rows
                if (b.size(0) != A_rows && A_rows > 0) {
                    b = b.index({torch::indexing::Slice(0, A_rows)});
                    if (b.size(0) < A_rows) {
                        // Pad if needed
                        auto padding = A_rows - b.size(0);
                        auto pad_tensor = torch::zeros({padding}, b.options());
                        b = torch::cat({b, pad_tensor}, 0);
                    }
                }
            } else if (b.dim() >= 2) {
                // For multi-dimensional b, make sure its second-to-last dimension matches A's rows
                if (b.size(-2) != A_rows && A_rows > 0) {
                    std::vector<torch::indexing::TensorIndex> indices(b.dim(), torch::indexing::Slice());
                    indices[b.dim() - 2] = torch::indexing::Slice(0, A_rows);
                    b = b.index(indices);
                    
                    if (b.size(-2) < A_rows) {
                        // Create a new tensor with the right size and copy data
                        std::vector<int64_t> new_sizes(b.sizes().begin(), b.sizes().end());
                        new_sizes[new_sizes.size() - 2] = A_rows;
                        auto new_b = torch::zeros(new_sizes, b.options());
                        
                        std::vector<torch::indexing::TensorIndex> src_indices(b.dim(), torch::indexing::Slice());
                        std::vector<torch::indexing::TensorIndex> dst_indices(b.dim(), torch::indexing::Slice());
                        dst_indices[b.dim() - 2] = torch::indexing::Slice(0, b.size(-2));
                        
                        new_b.index_put_(dst_indices, b.index(src_indices));
                        b = new_b;
                    }
                }
            }
        }
        
        // Apply triangular_solve
        auto result = torch::triangular_solve(b, A, upper, transpose, unitriangular);
        
        // Access the solution and LU factors to ensure they're computed
        auto solution = std::get<0>(result);
        auto A_clone = std::get<1>(result);
        
        // Perform some operations on the results to ensure they're valid
        auto sum = solution.sum();
        auto product = A_clone.matmul(solution);
        
        // Check if the solution is valid by comparing A * solution with b
        if (solution.numel() > 0 && A_clone.numel() > 0) {
            auto residual = product - b;
            auto norm = residual.norm().item<double>();
            
            // This is just to use the computed values and prevent dead code elimination
            if (std::isnan(norm) || std::isinf(norm)) {
                return 0;
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
