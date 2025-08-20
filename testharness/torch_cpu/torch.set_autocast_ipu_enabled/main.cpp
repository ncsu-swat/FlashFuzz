#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the enabled flag
        if (Size < 1) {
            return 0;
        }
        
        // Extract a boolean value from the first byte
        bool enabled = (Data[0] % 2) == 1;
        offset++;
        
        // Set autocast IPU enabled state
        torch::autocast::set_autocast_ipu_enabled(enabled);
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might be affected by autocast
            torch::Tensor result = tensor + 1.0;
            
            // Try a more complex operation
            if (tensor.dim() > 0 && tensor.size(0) > 0) {
                torch::Tensor matmul_result;
                try {
                    matmul_result = torch::matmul(tensor, tensor);
                } catch (const std::exception&) {
                    // Matmul might fail due to incompatible dimensions, that's fine
                }
            }
        }
        
        // Toggle the autocast state and try again with a different tensor
        torch::autocast::set_autocast_ipu_enabled(!enabled);
        
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform operations with the toggled autocast state
            torch::Tensor result2 = tensor2 * 2.0;
            
            // Try another operation
            torch::Tensor exp_result = torch::exp(tensor2);
        }
        
        // Reset to original state
        torch::autocast::set_autocast_ipu_enabled(enabled);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}