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
        
        // Create a square matrix for eigvals
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // eigvals requires a square matrix (n x n)
        // If tensor is not 2D, reshape it to a square matrix
        if (input.dim() != 2 || input.size(0) != input.size(1)) {
            // Calculate total number of elements
            int64_t total_elements = input.numel();
            
            // Determine the size of the square matrix
            int64_t square_size = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(total_elements)));
            
            // Resize the tensor to be square
            input = input.reshape({-1}).slice(0, 0, square_size * square_size);
            input = input.reshape({square_size, square_size});
        }
        
        // Convert to float or complex if needed for numerical stability
        if (input.scalar_type() == torch::kBool || 
            input.scalar_type() == torch::kUInt8 || 
            input.scalar_type() == torch::kInt8 || 
            input.scalar_type() == torch::kInt16 || 
            input.scalar_type() == torch::kInt32 || 
            input.scalar_type() == torch::kInt64) {
            input = input.to(torch::kFloat);
        }
        
        // Apply the eigvals operation
        torch::Tensor eigenvalues = torch::eigvals(input);
        
        // Optional: Test with different input types if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with potentially different type
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make it square
            if (input2.dim() != 2 || input2.size(0) != input2.size(1)) {
                int64_t total_elements = input2.numel();
                int64_t square_size = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(total_elements)));
                input2 = input2.reshape({-1}).slice(0, 0, square_size * square_size);
                input2 = input2.reshape({square_size, square_size});
            }
            
            // Convert to appropriate type if needed
            if (input2.scalar_type() == torch::kBool || 
                input2.scalar_type() == torch::kUInt8 || 
                input2.scalar_type() == torch::kInt8 || 
                input2.scalar_type() == torch::kInt16 || 
                input2.scalar_type() == torch::kInt32 || 
                input2.scalar_type() == torch::kInt64) {
                input2 = input2.to(torch::kFloat);
            }
            
            // Test eigvals on the second tensor
            torch::Tensor eigenvalues2 = torch::eigvals(input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}