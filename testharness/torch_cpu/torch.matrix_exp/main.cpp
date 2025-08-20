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
        
        // Create a square matrix for matrix_exp
        // matrix_exp requires a square matrix (n x n)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a square matrix (at least 2D tensor with last two dims equal)
        if (input.dim() >= 2) {
            // Get the last two dimensions
            int64_t last_dim = input.size(-1);
            int64_t second_last_dim = input.size(-2);
            
            // If not square, reshape to make it square
            if (last_dim != second_last_dim) {
                // Choose the smaller dimension as the square size
                int64_t square_size = std::min(last_dim, second_last_dim);
                
                // Create a new shape vector
                std::vector<int64_t> new_shape;
                for (int64_t i = 0; i < input.dim() - 2; ++i) {
                    new_shape.push_back(input.size(i));
                }
                new_shape.push_back(square_size);
                new_shape.push_back(square_size);
                
                // Reshape the tensor
                input = input.reshape(new_shape);
            }
            
            // Apply matrix_exp
            torch::Tensor result = torch::matrix_exp(input);
        } 
        else if (input.dim() == 1) {
            // For 1D tensor, reshape to 1x1 matrix
            input = input.reshape({1, 1});
            torch::Tensor result = torch::matrix_exp(input);
        }
        else if (input.dim() == 0) {
            // For scalar, reshape to 1x1 matrix
            input = input.reshape({1, 1});
            torch::Tensor result = torch::matrix_exp(input);
        }
        
        // Try with different data types if we have more data
        if (offset + 1 < Size) {
            // Create another tensor with potentially different dtype
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Make it square if it has at least 2 dimensions
            if (input2.dim() >= 2) {
                int64_t last_dim = input2.size(-1);
                int64_t second_last_dim = input2.size(-2);
                
                if (last_dim != second_last_dim) {
                    int64_t square_size = std::min(last_dim, second_last_dim);
                    
                    std::vector<int64_t> new_shape;
                    for (int64_t i = 0; i < input2.dim() - 2; ++i) {
                        new_shape.push_back(input2.size(i));
                    }
                    new_shape.push_back(square_size);
                    new_shape.push_back(square_size);
                    
                    input2 = input2.reshape(new_shape);
                }
                
                // Try matrix_exp with different dtype
                torch::Tensor result2 = torch::matrix_exp(input2);
            }
            else if (input2.dim() <= 1) {
                // Reshape to 1x1 matrix
                input2 = input2.reshape({1, 1});
                torch::Tensor result2 = torch::matrix_exp(input2);
            }
        }
        
        // Try with empty tensor if we have more data
        if (offset + 1 < Size) {
            // Create an empty tensor
            std::vector<int64_t> empty_shape = {0, 0};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            
            // Try matrix_exp with empty tensor
            try {
                torch::Tensor result_empty = torch::matrix_exp(empty_tensor);
            } catch (...) {
                // Expected to fail for empty tensor, just continue
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