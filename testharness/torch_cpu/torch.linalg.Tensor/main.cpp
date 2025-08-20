#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.linalg.Tensor operation
        // Note: torch.linalg.Tensor is just a type annotation in PyTorch
        // In C++ frontend, we're already working with torch::Tensor objects
        // So we'll test some common linear algebra operations instead
        
        // Test various linear algebra operations based on tensor properties
        if (input_tensor.dim() >= 2) {
            // For matrices, try matrix operations
            try {
                // Try matrix inverse if square
                if (input_tensor.size(0) == input_tensor.size(1)) {
                    auto inv = torch::inverse(input_tensor);
                }
            } catch (...) {
                // Ignore exceptions from the operations themselves
            }
            
            try {
                // Try SVD
                auto svd_result = torch::svd(input_tensor);
            } catch (...) {
                // Ignore exceptions from the operations themselves
            }
            
            try {
                // Try QR decomposition
                auto qr_result = torch::qr(input_tensor);
            } catch (...) {
                // Ignore exceptions from the operations themselves
            }
            
            try {
                // Try matrix norm
                auto norm = torch::norm(input_tensor);
            } catch (...) {
                // Ignore exceptions from the operations themselves
            }
        } else {
            // For vectors or scalars, try vector operations
            try {
                // Try vector norm
                auto norm = torch::norm(input_tensor);
            } catch (...) {
                // Ignore exceptions from the operations themselves
            }
        }
        
        // Try creating a diagonal matrix from the tensor
        try {
            auto diag = torch::diag(input_tensor);
        } catch (...) {
            // Ignore exceptions from the operations themselves
        }
        
        // Try matrix exponential if appropriate
        if (input_tensor.dim() == 2 && input_tensor.size(0) == input_tensor.size(1)) {
            try {
                auto matrix_exp = torch::matrix_exp(input_tensor);
            } catch (...) {
                // Ignore exceptions from the operations themselves
            }
        }
        
        // Try matrix power if appropriate
        if (input_tensor.dim() == 2 && input_tensor.size(0) == input_tensor.size(1)) {
            try {
                // Use a small power to avoid excessive computation
                int64_t power = Data[offset % Size] % 5;
                auto matrix_power = torch::matrix_power(input_tensor, power);
            } catch (...) {
                // Ignore exceptions from the operations themselves
            }
        }
        
        // Try determinant if appropriate
        if (input_tensor.dim() == 2 && input_tensor.size(0) == input_tensor.size(1)) {
            try {
                auto det = torch::det(input_tensor);
            } catch (...) {
                // Ignore exceptions from the operations themselves
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