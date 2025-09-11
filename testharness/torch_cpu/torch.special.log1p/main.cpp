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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.log1p operation
        torch::Tensor result = torch::special::log1p(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            result.item();
        }
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try in-place version if available
            torch::Tensor inplace_result = input2.clone();
            torch::special::log1p_out(inplace_result, input2);
            
            // Try with different output types
            if (offset + 1 < Size) {
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Try to cast input to different dtype
                torch::Tensor input_cast = input.to(dtype);
                torch::Tensor result_cast = torch::special::log1p(input_cast);
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
