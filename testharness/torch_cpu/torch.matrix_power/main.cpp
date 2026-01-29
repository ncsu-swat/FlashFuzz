#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least 2 bytes: 1 for exponent, 1 for tensor data
        if (Size < 2) {
            return 0;
        }
        
        // Reserve first byte for exponent
        int64_t n = static_cast<int8_t>(Data[0]);
        
        size_t offset = 1;
        
        // Create a tensor from remaining data
        torch::Tensor matrix = fuzzer_utils::createTensor(Data + 1, Size - 1, offset);
        
        // Ensure the tensor is a square matrix (2D with equal dimensions)
        // We need to carefully handle reshaping to maintain valid element counts
        int64_t total_elements = matrix.numel();
        
        if (total_elements == 0) {
            return 0;  // Can't create a valid matrix from empty tensor
        }
        
        // Calculate the largest square dimension we can make
        int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
        if (side == 0) side = 1;
        
        // Slice to get exactly side*side elements, then reshape
        int64_t needed_elements = side * side;
        torch::Tensor flat = matrix.flatten();
        
        if (flat.numel() < needed_elements) {
            // Pad with zeros if needed
            auto padding = torch::zeros({needed_elements - flat.numel()}, flat.options());
            flat = torch::cat({flat, padding});
        } else {
            flat = flat.slice(0, 0, needed_elements);
        }
        
        matrix = flat.reshape({side, side});
        
        // Convert to a supported dtype for matrix_power
        if (matrix.dtype() == torch::kBool || 
            matrix.dtype() == torch::kUInt8 || 
            matrix.dtype() == torch::kInt8 || 
            matrix.dtype() == torch::kInt16 ||
            matrix.dtype() == torch::kInt32 ||
            matrix.dtype() == torch::kInt64) {
            matrix = matrix.to(torch::kFloat);
        }
        
        // Inner try-catch for expected runtime errors (e.g., singular matrix for negative n)
        try {
            // Apply matrix_power operation using linalg API
            torch::Tensor result = torch::linalg_matrix_power(matrix, n);
            
            // Perform some operations on the result to ensure it's used
            auto sum = result.sum().item<double>();
            (void)sum;  // Suppress unused variable warning
        } catch (const std::exception &) {
            // Silently catch expected errors like singular matrix
        }
        
        // Also test the tensor method variant
        try {
            torch::Tensor result2 = matrix.matrix_power(n);
            auto sum2 = result2.sum().item<double>();
            (void)sum2;
        } catch (const std::exception &) {
            // Silently catch expected errors
        }
        
        // Test with batched matrices (3D input)
        if (Size > 10 && side >= 2) {
            try {
                // Create a batch of 2 matrices
                torch::Tensor batched = torch::stack({matrix, matrix * 0.5f});
                torch::Tensor batch_result = torch::linalg_matrix_power(batched, n);
                auto batch_sum = batch_result.sum().item<double>();
                (void)batch_sum;
            } catch (const std::exception &) {
                // Silently catch expected errors
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