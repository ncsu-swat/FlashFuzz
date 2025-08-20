#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for torch.special.zeta
        // zeta(x, q) requires two tensors
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset < Size) {
            torch::Tensor q = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply torch.special.zeta operation
            torch::Tensor result = torch::special::zeta(x, q);
            
            // Try the scalar version as well if possible
            if (x.numel() > 0) {
                // Get a scalar value from the tensor
                auto scalar_x = x.item();
                torch::Tensor result_scalar_x = torch::special::zeta(scalar_x, q);
            }
            
            if (q.numel() > 0) {
                // Get a scalar value from the tensor
                auto scalar_q = q.item();
                torch::Tensor result_scalar_q = torch::special::zeta(x, scalar_q);
            }
        } else {
            // If we don't have enough data for a second tensor, use default value of 1
            torch::Tensor ones = torch::ones_like(x);
            torch::Tensor result = torch::special::zeta(x, ones);
            
            // Try the scalar version as well if possible
            if (x.numel() > 0) {
                auto scalar_x = x.item();
                torch::Tensor result_scalar = torch::special::zeta(scalar_x, ones);
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