#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the boolean flag
        if (Size < 1) {
            return 0;
        }
        
        // Extract a boolean value from the first byte
        bool enabled = Data[0] & 0x1;
        offset++;
        
        // Set autocast enabled state
        torch::set_autocast_enabled(enabled);
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might be affected by autocast
            torch::Tensor result = tensor + tensor;
            
            // Try matrix multiplication if tensor has at least 2 dimensions
            if (tensor.dim() >= 2) {
                try {
                    torch::Tensor matmul_result = torch::matmul(tensor, tensor);
                } catch (const std::exception&) {
                    // Ignore exceptions from matmul (e.g., shape mismatch)
                }
            }
            
            // Try some other operations that might be affected by autocast
            torch::Tensor sin_result = torch::sin(tensor);
            torch::Tensor exp_result = torch::exp(tensor);
            
            // Reset autocast state to default (false)
            torch::set_autocast_enabled(false);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}