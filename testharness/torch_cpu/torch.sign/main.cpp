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
        
        // Apply torch.sign operation
        torch::Tensor result = torch::sign(input);
        
        // Try some variations if we have more data
        if (offset + 1 < Size) {
            // Try out_variant if we have more data
            torch::Tensor out = torch::empty_like(input);
            torch::sign_out(out, input);
            
            // Try in-place variant if we have more data
            torch::Tensor inplace = input.clone();
            inplace.sign_();
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor result2 = torch::sign(input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
