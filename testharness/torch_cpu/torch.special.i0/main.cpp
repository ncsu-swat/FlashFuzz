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
        
        // Create input tensor for torch.special.i0
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.i0 operation
        torch::Tensor result = torch::special::i0(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation of the tensor
            if (result.numel() > 0) {
                result.item();
            }
        }
        
        // Try with out variant if we have enough data left
        if (offset + 2 < Size) {
            // Create output tensor with same shape and dtype as input
            torch::Tensor output = torch::empty_like(input);
            
            // Apply torch.special.i0 with out parameter
            torch::special::i0_out(output, input);
            
            // Force evaluation
            if (output.numel() > 0) {
                output.item();
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