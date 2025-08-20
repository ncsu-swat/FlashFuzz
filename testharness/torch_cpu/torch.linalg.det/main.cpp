#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a square matrix for determinant calculation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Determinant requires a square matrix (n x n)
        // If tensor is not 2D, reshape it to a square matrix if possible
        if (input.dim() != 2 || input.size(0) != input.size(1)) {
            // Calculate total number of elements
            int64_t total_elements = input.numel();
            
            // Find a square size that can fit in the tensor
            int64_t square_size = static_cast<int64_t>(std::sqrt(total_elements));
            
            // Reshape to square matrix
            if (square_size > 0) {
                input = input.reshape({square_size, square_size});
            } else {
                // If we can't reshape to a square matrix, create a small one
                input = torch::ones({1, 1}, input.options());
            }
        }
        
        // Try to convert to a supported dtype for determinant if needed
        if (input.scalar_type() == torch::kBool || 
            input.scalar_type() == torch::kUInt8 || 
            input.scalar_type() == torch::kInt8 || 
            input.scalar_type() == torch::kInt16 || 
            input.scalar_type() == torch::kInt32 || 
            input.scalar_type() == torch::kInt64) {
            input = input.to(torch::kFloat);
        }
        
        // Calculate determinant
        torch::Tensor det = torch::det(input);
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Edge case: matrix with zeros
            if (edge_case % 5 == 0) {
                torch::Tensor zero_matrix = torch::zeros_like(input);
                torch::Tensor zero_det = torch::det(zero_matrix);
            }
            
            // Edge case: identity matrix
            if (edge_case % 5 == 1) {
                torch::Tensor identity = torch::eye(input.size(0), input.options());
                torch::Tensor identity_det = torch::det(identity);
            }
            
            // Edge case: matrix with very large values
            if (edge_case % 5 == 2) {
                torch::Tensor large_matrix = input * 1e10;
                torch::Tensor large_det = torch::det(large_matrix);
            }
            
            // Edge case: matrix with very small values
            if (edge_case % 5 == 3) {
                torch::Tensor small_matrix = input * 1e-10;
                torch::Tensor small_det = torch::det(small_matrix);
            }
            
            // Edge case: singular matrix (if possible)
            if (edge_case % 5 == 4 && input.size(0) > 1) {
                torch::Tensor singular = input.clone();
                // Make one row a multiple of another to create singularity
                singular.index_put_({1}, singular.index({0}));
                torch::Tensor singular_det = torch::det(singular);
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