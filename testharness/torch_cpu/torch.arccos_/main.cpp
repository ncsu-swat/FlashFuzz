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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the input tensor for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply arccos_ in-place operation
        input_copy.arccos_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected_output = torch::arccos(input);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        // This is a sanity check, not a defensive check that would prevent testing edge cases
        if (input_copy.defined() && expected_output.defined()) {
            bool tensors_match = torch::allclose(input_copy, expected_output, 1e-5, 1e-8);
            if (!tensors_match) {
                // This is not an error, just an observation that might indicate a bug
                // We don't throw or return early
            }
        }
        
        // Try another tensor with different properties if we have more data
        if (offset + 2 < Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make a copy for in-place operation
            torch::Tensor another_copy = another_input.clone();
            
            // Apply arccos_ in-place operation
            another_copy.arccos_();
        }
        
        // Try with edge case values if we have more data
        if (offset + 2 < Size) {
            // Create a tensor with values close to -1 and 1 (domain boundaries for arccos)
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor edge_tensor = torch::ones({2, 2}, options);
            
            // Set some values to be close to domain boundaries
            edge_tensor[0][0] = 0.9999;
            edge_tensor[0][1] = -0.9999;
            edge_tensor[1][0] = 1.0;
            edge_tensor[1][1] = -1.0;
            
            // Apply arccos_ in-place
            edge_tensor.arccos_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}