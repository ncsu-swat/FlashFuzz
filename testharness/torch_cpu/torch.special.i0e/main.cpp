#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.special.i0e
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the torch.special.i0e operation
        torch::Tensor result = torch::special::i0e(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation of the tensor
            if (result.numel() > 0) {
                result.item();
            }
        }
        
        // Try with different input configurations if we have more data
        if (Size - offset > 2) {
            // Create another tensor with different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply the operation again
            torch::Tensor result2 = torch::special::i0e(input2);
            
            // Force evaluation
            if (result2.defined() && result2.numel() > 0) {
                result2.item();
            }
        }
        
        // Test with edge cases if we have enough data
        if (Size - offset > 2) {
            // Create a tensor with potentially extreme values
            torch::Tensor edge_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply special scaling or transformations to create edge cases
            torch::Tensor scaled_input = edge_input * 1e10;
            torch::Tensor result3 = torch::special::i0e(scaled_input);
            
            if (result3.defined() && result3.numel() > 0) {
                result3.item();
            }
            
            // Test with negative values
            torch::Tensor neg_input = -edge_input;
            torch::Tensor result4 = torch::special::i0e(neg_input);
            
            if (result4.defined() && result4.numel() > 0) {
                result4.item();
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