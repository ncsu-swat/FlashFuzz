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
        
        // Create a square matrix tensor
        torch::Tensor matrix = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to make it a square matrix for operations that require it
        if (matrix.dim() >= 2) {
            int64_t min_dim = std::min(matrix.size(0), matrix.size(1));
            matrix = matrix.slice(0, 0, min_dim).slice(1, 0, min_dim);
        } else if (matrix.dim() == 1) {
            // Convert 1D tensor to 2D square matrix
            int64_t size = matrix.size(0);
            matrix = matrix.reshape({size, 1}).expand({size, size});
        } else if (matrix.dim() == 0) {
            // Convert scalar to 1x1 matrix
            matrix = matrix.reshape({1, 1});
        }
        
        // Try to force a singular matrix to trigger LinAlgError
        if (matrix.dim() >= 2 && matrix.size(0) > 1 && matrix.size(1) > 1) {
            // Make a column or row all zeros to create a singular matrix
            if (offset < Size) {
                uint8_t idx = Data[offset++] % matrix.size(0);
                matrix.index_put_({idx, torch::indexing::Slice()}, 0);
            }
        }
        
        // Convert to float for numerical operations
        if (matrix.dtype() != torch::kFloat && 
            matrix.dtype() != torch::kDouble && 
            matrix.dtype() != torch::kComplexFloat && 
            matrix.dtype() != torch::kComplexDouble) {
            matrix = matrix.to(torch::kFloat);
        }
        
        // Try operations that might throw LinAlgError
        try {
            // Try matrix inverse (will fail for singular matrices)
            torch::Tensor inverse = torch::inverse(matrix);
        } catch (const torch::Error& e) {
            // Check if it's a LinAlgError
            std::string error_msg = e.what();
            if (error_msg.find("singular") != std::string::npos || 
                error_msg.find("LinAlgError") != std::string::npos) {
                // Successfully triggered LinAlgError
            }
        }
        
        try {
            // Try Cholesky decomposition (requires positive definite matrix)
            torch::Tensor cholesky = torch::cholesky(matrix);
        } catch (const torch::Error& e) {
            // Check if it's a LinAlgError
            std::string error_msg = e.what();
            if (error_msg.find("not positive definite") != std::string::npos || 
                error_msg.find("LinAlgError") != std::string::npos) {
                // Successfully triggered LinAlgError
            }
        }
        
        try {
            // Try LU decomposition with singular matrix
            auto lu_result = torch::lu(matrix);
        } catch (const torch::Error& e) {
            // Check if it's a LinAlgError
            std::string error_msg = e.what();
            if (error_msg.find("singular") != std::string::npos || 
                error_msg.find("LinAlgError") != std::string::npos) {
                // Successfully triggered LinAlgError
            }
        }
        
        try {
            // Try solving linear system with singular matrix
            torch::Tensor b;
            if (matrix.dim() >= 2) {
                b = torch::ones({matrix.size(0), 1}, matrix.options());
                torch::Tensor solution = torch::solve(b, matrix);
            }
        } catch (const torch::Error& e) {
            // Check if it's a LinAlgError
            std::string error_msg = e.what();
            if (error_msg.find("singular") != std::string::npos || 
                error_msg.find("LinAlgError") != std::string::npos) {
                // Successfully triggered LinAlgError
            }
        }
        
        try {
            // Try eigenvalue decomposition with problematic inputs
            if (matrix.dim() >= 2 && matrix.size(0) == matrix.size(1)) {
                auto eigenvalues = torch::eig(matrix);
            }
        } catch (const torch::Error& e) {
            // Check if it's a LinAlgError
            std::string error_msg = e.what();
            if (error_msg.find("LinAlgError") != std::string::npos) {
                // Successfully triggered LinAlgError
            }
        }
        
        try {
            // Try SVD with problematic inputs
            auto svd_result = torch::svd(matrix);
        } catch (const torch::Error& e) {
            // Check if it's a LinAlgError
            std::string error_msg = e.what();
            if (error_msg.find("LinAlgError") != std::string::npos) {
                // Successfully triggered LinAlgError
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
