#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/autograd/autocast_mode.h>

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
        
        // Set the autocast XLA enabled flag
        torch::autograd::set_autocast_xla_enabled(enabled);
        
        // Verify the setting was applied correctly
        bool current_setting = torch::autograd::is_autocast_xla_enabled();
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test autocast with the tensor
            {
                torch::autograd::AutocastMode autocast_mode(at::kCUDA);
                
                // Perform some operations that might be affected by autocast
                torch::Tensor result = tensor + tensor;
                result = torch::matmul(result, result);
                
                // Test with different device types if available
                if (torch::cuda::is_available()) {
                    torch::Tensor cuda_tensor = tensor.cuda();
                    torch::Tensor cuda_result = cuda_tensor + cuda_tensor;
                }
            }
        }
        
        // Toggle the setting and test again
        torch::autograd::set_autocast_xla_enabled(!enabled);
        
        // Create another tensor with different settings
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test with the new setting
            {
                torch::autograd::AutocastMode autocast_mode(at::kCUDA);
                torch::Tensor result = tensor2 * 2.0;
            }
        }
        
        // Restore the original setting
        torch::autograd::set_autocast_xla_enabled(enabled);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}