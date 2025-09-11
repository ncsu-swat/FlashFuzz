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
        
        // Need at least some data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for lu_factor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // lu_factor requires a square matrix (n x n) or a batch of square matrices
        // We'll try to handle various cases including edge cases
        
        // Get the tensor dimensions
        auto dims = A.sizes().vec();
        
        // If tensor is 0-dim or 1-dim, reshape it to a square matrix
        if (dims.size() <= 1) {
            int64_t total_elements = A.numel();
            int64_t matrix_size = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(total_elements)));
            
            // Pad with zeros if needed
            if (matrix_size * matrix_size > total_elements) {
                A = torch::constant_pad_nd(A.reshape(-1), {0, matrix_size * matrix_size - total_elements}, 0);
            }
            
            A = A.reshape({matrix_size, matrix_size});
        }
        // For 2+ dimensional tensors, ensure the last two dimensions are square
        else {
            auto last_dim = dims.back();
            auto second_last_dim = dims[dims.size() - 2];
            
            // If the last two dimensions aren't equal, make them equal
            if (last_dim != second_last_dim) {
                int64_t square_size = std::max(last_dim, second_last_dim);
                
                // Create new shape with square matrix in last two dimensions
                std::vector<int64_t> new_shape = dims;
                new_shape[dims.size() - 2] = square_size;
                new_shape[dims.size() - 1] = square_size;
                
                // Resize tensor with padding if needed
                A = torch::constant_pad_nd(A, {0, square_size - last_dim, 0, square_size - second_last_dim}, 0);
                A = A.reshape(new_shape);
            }
        }
        
        // Apply lu_factor operation
        auto result = torch::lu(A, true);
        
        // Unpack the result to verify it works
        auto& LU = std::get<0>(result);
        auto& pivots = std::get<1>(result);
        
        // Perform some basic operations on the results to ensure they're valid
        auto LU_sum = LU.sum();
        auto pivots_sum = pivots.sum();
        
        // Prevent compiler from optimizing away the operations
        if (LU_sum.item<double>() == std::numeric_limits<double>::infinity() && 
            pivots_sum.item<int64_t>() == std::numeric_limits<int64_t>::max()) {
            throw std::runtime_error("Unexpected infinite values");
        }
        
        // Try reconstructing the original matrix to test the full functionality
        if (A.dim() == 2) {
            try {
                auto P = torch::eye(A.size(0), A.options().dtype(torch::kFloat));
                for (int64_t i = 0; i < pivots.size(0); i++) {
                    int64_t pivot = pivots[i].item<int64_t>() - 1; // PyTorch uses 1-based indexing for pivots
                    if (i != pivot && pivot >= 0 && pivot < A.size(0)) {
                        auto temp = P.index({i}).clone();
                        P.index_put_({i}, P.index({pivot}));
                        P.index_put_({pivot}, temp);
                    }
                }
                
                // Extract L and U from LU
                auto L = torch::tril(LU, -1) + torch::eye(LU.size(0), LU.size(1), LU.options());
                auto U = torch::triu(LU);
                
                // Compute P^-1 * L * U to check if it equals A
                auto reconstructed = torch::matmul(torch::matmul(P.transpose(0, 1), L), U);
                
                // Verify reconstruction (this might throw if there are numerical issues)
                auto diff = torch::abs(reconstructed - A.to(reconstructed.dtype())).max().item<double>();
                if (std::isnan(diff) || std::isinf(diff)) {
                    throw std::runtime_error("NaN or Inf in reconstruction difference");
                }
            } catch (const std::exception& e) {
                // Reconstruction might fail for singular matrices, which is expected
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
