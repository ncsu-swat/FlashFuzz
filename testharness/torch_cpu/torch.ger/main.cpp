#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two input vectors for torch.ger
        // ger requires two 1D tensors (vectors)
        torch::Tensor vec1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have more data, create the second tensor
        if (offset < Size) {
            torch::Tensor vec2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to reshape tensors to 1D if they're not already
            // This helps test more cases without excessive defensive checks
            if (vec1.dim() != 1) {
                vec1 = vec1.reshape({-1});
            }
            
            if (vec2.dim() != 1) {
                vec2 = vec2.reshape({-1});
            }
            
            // Apply the ger operation
            // ger(input, vec2) → Tensor
            // Outer product of input and vec2
            torch::Tensor result = torch::ger(vec1, vec2);
            
            // Verify the result has the expected shape
            // If vec1 has size m and vec2 has size n, result should be m×n
            if (result.dim() == 2 && 
                result.size(0) == vec1.size(0) && 
                result.size(1) == vec2.size(0)) {
                // Basic shape verification passed
            }
            
            // Test some edge cases with empty tensors
            if (offset + 2 < Size) {
                // Create empty tensors for edge case testing
                torch::Tensor empty_vec = torch::empty({0}, vec1.options());
                
                // Try ger with an empty tensor
                try {
                    torch::Tensor empty_result1 = torch::ger(empty_vec, vec2);
                } catch (...) {
                    // Expected exception for some cases
                }
                
                try {
                    torch::Tensor empty_result2 = torch::ger(vec1, empty_vec);
                } catch (...) {
                    // Expected exception for some cases
                }
                
                // Try with both empty
                try {
                    torch::Tensor empty_result3 = torch::ger(empty_vec, empty_vec);
                } catch (...) {
                    // Expected exception for some cases
                }
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