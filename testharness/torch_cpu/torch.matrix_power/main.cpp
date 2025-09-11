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
        
        // Need at least 1 byte for the exponent
        if (Size < 1) {
            return 0;
        }
        
        // Create a square matrix tensor
        torch::Tensor matrix;
        if (offset < Size) {
            matrix = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure the tensor is a square matrix (2D with equal dimensions)
            if (matrix.dim() == 2) {
                int64_t n = matrix.size(0);
                if (matrix.size(1) != n) {
                    // Reshape to make it square if needed
                    matrix = matrix.reshape({n, n});
                }
            } else if (matrix.dim() == 0) {
                // Scalar - reshape to 1x1 matrix
                matrix = matrix.reshape({1, 1});
            } else if (matrix.dim() == 1) {
                // 1D tensor - reshape to square matrix
                int64_t n = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(matrix.numel())));
                matrix = matrix.reshape({n, n});
            } else {
                // Higher dimensional tensor - flatten and reshape to square matrix
                int64_t total_elements = matrix.numel();
                int64_t n = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(total_elements)));
                matrix = matrix.reshape({n, n});
            }
            
            // Convert to a supported dtype for matrix_power if needed
            if (matrix.dtype() == torch::kBool || 
                matrix.dtype() == torch::kUInt8 || 
                matrix.dtype() == torch::kInt8 || 
                matrix.dtype() == torch::kInt16) {
                matrix = matrix.to(torch::kFloat);
            }
        } else {
            // Create a default small matrix if we don't have enough data
            matrix = torch::eye(2);
        }
        
        // Get the exponent from the input data
        int64_t n = 0;
        if (offset < Size) {
            // Use the next byte as exponent
            n = static_cast<int8_t>(Data[offset++]);
        }
        
        // Apply matrix_power operation
        torch::Tensor result = torch::linalg_matrix_power(matrix, n);
        
        // Try alternative API
        if (offset < Size && Data[offset] % 2 == 0) {
            result = matrix.matrix_power(n);
        }
        
        // Perform some operations on the result to ensure it's used
        auto sum = result.sum().item<double>();
        if (std::isnan(sum) || std::isinf(sum)) {
            // Just to use the result, not throwing
            result = torch::zeros_like(result);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
