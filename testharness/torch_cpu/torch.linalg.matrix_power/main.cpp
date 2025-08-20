#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation and power parameter
        if (Size < 4) {
            return 0;
        }
        
        // Create a square matrix tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a square matrix (at least 2D tensor with equal last two dimensions)
        if (A.dim() < 2) {
            // If tensor is less than 2D, reshape it to a square matrix
            int64_t elements = A.numel();
            int64_t dim_size = static_cast<int64_t>(std::sqrt(elements));
            if (dim_size > 0) {
                A = A.reshape({dim_size, dim_size});
            } else {
                // Handle empty tensor case
                A = torch::zeros({1, 1}, A.options());
            }
        } else {
            // If tensor is at least 2D, make the last two dimensions equal
            std::vector<int64_t> shape = A.sizes().vec();
            int64_t last_dim = shape.back();
            shape[shape.size() - 2] = last_dim;
            A = A.reshape(shape);
        }
        
        // Extract n (power) from the input data
        int64_t n = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply matrix_power operation
        torch::Tensor result = torch::matrix_power(A, n);
        
        // Try different variations of the API
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            if (variant % 3 == 0) {
                // Test with named arguments
                result = torch::matrix_power(A, n);
            } else if (variant % 3 == 1) {
                // Test with out parameter - matrix_power doesn't have out variant, so just call normally
                result = torch::matrix_power(A, n);
            } else {
                // Test with different data type
                if (A.scalar_type() != torch::kDouble) {
                    A = A.to(torch::kDouble);
                    result = torch::matrix_power(A, n);
                } else {
                    A = A.to(torch::kFloat);
                    result = torch::matrix_power(A, n);
                }
            }
        }
        
        // Test edge cases with specific powers
        if (offset < Size) {
            uint8_t power_case = Data[offset++];
            switch (power_case % 5) {
                case 0:
                    // Power 0 (should return identity matrix)
                    result = torch::matrix_power(A, 0);
                    break;
                case 1:
                    // Power 1 (should return A)
                    result = torch::matrix_power(A, 1);
                    break;
                case 2:
                    // Negative power
                    result = torch::matrix_power(A, -1);
                    break;
                case 3:
                    // Large positive power
                    result = torch::matrix_power(A, 10);
                    break;
                case 4:
                    // Large negative power
                    result = torch::matrix_power(A, -10);
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