#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a diagonal or construct a diagonal matrix
        if (offset + 1 < Size) {
            // Get a value for diagonal offset parameter
            int64_t diagonal = static_cast<int8_t>(Data[offset++]);
            
            // Test torch::diag with the input tensor
            // diag requires 1D or 2D tensor, so wrap in try-catch for shape errors
            try {
                torch::Tensor result = torch::diag(input_tensor, diagonal);
                
                // Test edge case: apply diag to the result again
                if (offset < Size && Data[offset++] % 2 == 0) {
                    int64_t second_diagonal = static_cast<int8_t>(Data[offset++] % 10);
                    try {
                        torch::Tensor second_result = torch::diag(result, second_diagonal);
                    } catch (...) {
                        // Expected for certain shapes/diagonal combinations
                    }
                }
            } catch (...) {
                // Expected for non 1D/2D tensors or invalid diagonal
            }
        } else {
            // If we don't have enough data for the diagonal parameter,
            // just use the default diagonal (0)
            try {
                torch::Tensor result = torch::diag(input_tensor);
            } catch (...) {
                // Expected for non 1D/2D tensors
            }
        }
        
        // Test with different diagonal values if we have more data
        if (offset + 4 <= Size) {
            int64_t large_diagonal = static_cast<int32_t>(
                (Data[offset]) | (Data[offset+1] << 8) | 
                (Data[offset+2] << 16) | (Data[offset+3] << 24));
            offset += 4;
            
            // Try with a potentially large diagonal value
            try {
                torch::Tensor result_large_diag = torch::diag(input_tensor, large_diagonal);
            } catch (...) {
                // Expected for invalid diagonal or tensor shape
            }
        }
        
        // Test with negative diagonal values
        if (offset < Size) {
            int64_t negative_diagonal = -static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
            
            // Try with a negative diagonal value
            try {
                torch::Tensor result_neg_diag = torch::diag(input_tensor, negative_diagonal);
            } catch (...) {
                // Expected for invalid diagonal or tensor shape
            }
        }
        
        // Test with 1D tensor explicitly (valid input for diag)
        if (offset + 1 < Size) {
            int64_t vec_size = (Data[offset++] % 16) + 1; // 1 to 16 elements
            torch::Tensor vec_tensor = torch::randn({vec_size});
            int64_t diag_offset = static_cast<int8_t>(Data[offset++]);
            torch::Tensor diag_matrix = torch::diag(vec_tensor, diag_offset);
        }
        
        // Test with 2D tensor explicitly (extract diagonal)
        if (offset + 2 < Size) {
            int64_t rows = (Data[offset++] % 8) + 1; // 1 to 8
            int64_t cols = (Data[offset++] % 8) + 1; // 1 to 8
            torch::Tensor mat_tensor = torch::randn({rows, cols});
            int64_t diag_offset = static_cast<int8_t>(Data[offset++] % 10) - 5; // -5 to 4
            try {
                torch::Tensor extracted_diag = torch::diag(mat_tensor, diag_offset);
            } catch (...) {
                // Expected for out of bounds diagonal
            }
        }
        
        // Test with empty tensor
        if (offset < Size && Data[offset++] % 3 == 0) {
            torch::Tensor empty_1d = torch::empty({0});
            try {
                torch::Tensor empty_result = torch::diag(empty_1d);
            } catch (...) {
                // May throw for empty tensor
            }
        }
        
        // Test diag_embed for better coverage of diagonal operations
        if (offset + 2 < Size) {
            int64_t vec_len = (Data[offset++] % 8) + 1;
            torch::Tensor vec = torch::randn({vec_len});
            int64_t diag_offset = static_cast<int8_t>(Data[offset++]);
            try {
                torch::Tensor embedded = torch::diag_embed(vec, diag_offset);
            } catch (...) {
                // Expected for certain configurations
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}