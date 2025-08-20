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
        
        // Check if autocast is enabled (should be false by default)
        bool is_enabled_default = torch::autocast::is_enabled();
        
        // Create a tensor to use in autocast context
        if (Size > 0) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test autocast in different contexts
            {
                // Enable autocast
                torch::autocast::set_enabled(true);
                bool is_enabled_true = torch::autocast::is_enabled();
                
                // Perform some operation with the tensor in autocast context
                torch::Tensor result = tensor + 1.0;
                
                // Disable autocast
                torch::autocast::set_enabled(false);
                bool is_enabled_false = torch::autocast::is_enabled();
                
                // Perform operation again with autocast disabled
                torch::Tensor result2 = tensor + 1.0;
            }
            
            // Test nested autocast contexts
            {
                torch::autocast::set_enabled(true);
                bool outer_enabled = torch::autocast::is_enabled();
                
                {
                    // Nested context with different setting
                    torch::autocast::set_enabled(false);
                    bool inner_enabled = torch::autocast::is_enabled();
                }
                
                // Check if outer context is restored
                bool after_nested = torch::autocast::is_enabled();
                
                // Reset to default
                torch::autocast::set_enabled(false);
            }
            
            // Test with specific device type
            if (offset + 1 <= Size) {
                bool use_cuda = Data[offset++] % 2 == 0;
                
                if (use_cuda && torch::cuda::is_available()) {
                    torch::autocast::set_enabled(true);
                    bool cuda_enabled = torch::autocast::is_enabled();
                    torch::autocast::set_enabled(false);
                }
                
                // Test with CPU device
                torch::autocast::set_enabled(true);
                bool cpu_enabled = torch::autocast::is_enabled();
                torch::autocast::set_enabled(false);
            }
        }
        
        // Test with different device types if available
        if (offset + 1 <= Size) {
            uint8_t device_selector = Data[offset++];
            c10::DeviceType device_type;
            
            switch (device_selector % 3) {
                case 0:
                    device_type = torch::kCPU;
                    break;
                case 1:
                    device_type = torch::kCUDA;
                    break;
                case 2:
                    device_type = torch::kMPS;
                    break;
            }
            
            // Check if autocast is enabled for the selected device
            bool is_enabled_for_device = torch::autocast::is_enabled();
            
            // Enable autocast for the device and check again
            torch::autocast::set_enabled(true);
            bool is_enabled_after = torch::autocast::is_enabled();
            
            // Disable and check
            torch::autocast::set_enabled(false);
            bool is_disabled_after = torch::autocast::is_enabled();
        }
        
        // Ensure we reset autocast state to default
        torch::autocast::set_enabled(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}