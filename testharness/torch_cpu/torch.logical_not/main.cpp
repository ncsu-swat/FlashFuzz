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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply logical_not operation
        torch::Tensor result = torch::logical_not(input_tensor);
        
        // Try some variations if we have enough data
        if (offset + 1 < Size) {
            // Try in-place version
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.logical_not_();
            
            // Try with out parameter
            torch::Tensor out_tensor = torch::empty_like(input_tensor, input_tensor.options().dtype(torch::kBool));
            torch::logical_not_out(out_tensor, input_tensor);
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply logical_not to this tensor too
            torch::Tensor another_result = torch::logical_not(another_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}