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
        
        // Create input tensor for digamma operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the digamma operation
        torch::Tensor result = torch::special::digamma(input);
        
        // Try some variants of the operation
        if (offset + 1 < Size) {
            // If we have more data, try out_variant
            torch::Tensor out = torch::empty_like(input);
            torch::special::digamma_out(out, input);
            
            // Try in-place variant if supported for the dtype
            if (input.is_floating_point()) {
                torch::Tensor input_copy = input.clone();
                torch::digamma_(input_copy);
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