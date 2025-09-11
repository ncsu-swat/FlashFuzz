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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.exp2 operation
        torch::Tensor result = torch::special::exp2(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            result.item();
        }
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Test inplace version if available
            torch::Tensor input2_copy = input2.clone();
            torch::special::exp2_out(input2_copy, input2);
            
            // Test with different output types
            if (input.is_floating_point()) {
                // Test with double precision output
                torch::Tensor double_result = torch::special::exp2(input.to(torch::kDouble));
                
                // Test with half precision output if supported
                if (torch::cuda::is_available()) {
                    torch::Tensor half_result = torch::special::exp2(input.to(torch::kHalf));
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
