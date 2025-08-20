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
        
        // Create input tensor for torch.special.ndtr
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.ndtr operation
        torch::Tensor result = torch::special::ndtr(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto item = result.item();
        }
        
        // Try with different input configurations if we have more data
        if (Size - offset >= 2) {
            // Create another tensor with different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply torch.special.ndtr with the new tensor
            torch::Tensor result2 = torch::special::ndtr(input2);
            
            // Try to access the result
            if (result2.defined() && result2.numel() > 0) {
                auto item2 = result2.item();
            }
        }
        
        // Test with edge cases if we have more data
        if (Size - offset >= 2) {
            // Create a tensor with potentially extreme values
            torch::Tensor edge_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply operation to edge case
            torch::Tensor edge_result = torch::special::ndtr(edge_input);
            
            // Try to access the result
            if (edge_result.defined() && edge_result.numel() > 0) {
                auto edge_item = edge_result.item();
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