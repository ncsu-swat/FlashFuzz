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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply fliplr operation
        // fliplr reverses the order of elements along dimension 1 (columns)
        // It requires a tensor with at least 2 dimensions
        torch::Tensor result;
        
        // Try to apply fliplr regardless of tensor dimensions
        // Let PyTorch handle any errors for invalid inputs
        result = torch::fliplr(input_tensor);
        
        // Verify the operation worked correctly by checking properties
        if (result.defined()) {
            // Basic validation: shapes should match except for the flipped dimension
            if (input_tensor.dim() >= 2) {
                // For tensors with at least 2 dimensions, the shapes should be identical
                if (result.sizes() != input_tensor.sizes()) {
                    throw std::runtime_error("Shape mismatch after fliplr");
                }
            }
            
            // Verify the operation actually flipped the tensor
            // For a tensor with at least 2 dimensions, we can check if elements are reversed along dim 1
            if (input_tensor.dim() >= 2 && input_tensor.size(1) > 1) {
                // Check first and last columns if they match the expected flipped pattern
                // This is just a basic sanity check
                auto first_col_input = input_tensor.select(1, 0);
                auto last_col_input = input_tensor.select(1, input_tensor.size(1) - 1);
                
                auto first_col_result = result.select(1, 0);
                auto last_col_result = result.select(1, result.size(1) - 1);
                
                // First column of result should match last column of input
                if (!torch::allclose(first_col_result, last_col_input)) {
                    throw std::runtime_error("First column of result doesn't match last column of input");
                }
                
                // Last column of result should match first column of input
                if (!torch::allclose(last_col_result, first_col_input)) {
                    throw std::runtime_error("Last column of result doesn't match first column of input");
                }
            }
        }
        
        // Try to create another tensor if there's data left
        if (offset < Size - 2) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try fliplr on this tensor too
            torch::Tensor another_result = torch::fliplr(another_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
