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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.psi operation
        torch::Tensor result = torch::special::psi(input);
        
        // Try some variants with optional parameters
        if (offset + 1 < Size) {
            // Use the next byte to decide whether to test the out variant
            uint8_t test_out = Data[offset++];
            if (test_out % 2 == 0) {
                torch::Tensor out = torch::empty_like(input);
                torch::special::psi_out(out, input);
            }
        }
        
        // Try to access result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto item = result.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
