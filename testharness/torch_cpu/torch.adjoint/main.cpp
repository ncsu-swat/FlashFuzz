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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.adjoint operation
        // The adjoint operation requires a tensor with at least 2 dimensions
        // where the last two dimensions form a matrix
        torch::Tensor result;
        
        // Apply the adjoint operation
        // This will conjugate and transpose the matrix
        result = torch::adjoint(input_tensor);
        
        // Verify the result is not empty
        if (result.numel() > 0) {
            // Access some elements to ensure computation is performed
            if (result.dim() > 0 && result.size(0) > 0) {
                auto first_element = result.index({0});
            }
        }
        
        // Try another variant with different input if we have more data
        if (offset < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor another_result = torch::adjoint(another_tensor);
            
            // Verify the result tensor
            if (another_result.numel() > 0 && another_result.dim() > 0 && another_result.size(0) > 0) {
                auto first_element = another_result.index({0});
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