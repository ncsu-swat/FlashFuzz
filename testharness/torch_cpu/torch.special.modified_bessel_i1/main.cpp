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
        
        // Apply the modified_bessel_i1 operation
        torch::Tensor result = torch::special::modified_bessel_i1(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            result.item();
        }
        
        // Try with out variant if we have enough data
        if (offset + 1 < Size) {
            // Create output tensor with same shape and dtype as input
            torch::Tensor out = torch::empty_like(input);
            
            // Apply the operation with out parameter
            torch::special::modified_bessel_i1_out(out, input);
            
            // Access the result
            if (out.defined() && out.numel() > 0) {
                out.item();
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
