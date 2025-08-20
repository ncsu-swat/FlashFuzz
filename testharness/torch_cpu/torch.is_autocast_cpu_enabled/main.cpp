#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if autocast is enabled for CPU
        bool is_enabled = torch::autocast::is_cpu_enabled();
        
        // Try toggling autocast state
        if (Size > 0) {
            bool enable_state = Data[offset] % 2 == 0;
            offset++;
            
            // Set autocast state
            torch::autocast::set_cpu_enabled(enable_state);
            
            // Verify the state was set correctly
            bool new_state = torch::autocast::is_cpu_enabled();
            if (new_state != enable_state) {
                throw std::runtime_error("Autocast state not set correctly");
            }
        }
        
        // Create a tensor to test with autocast
        if (Size > offset) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test operations with autocast enabled/disabled
            torch::Tensor result;
            
            // Enable autocast
            torch::autocast::set_cpu_enabled(true);
            result = tensor * 2.0;
            
            // Disable autocast
            torch::autocast::set_cpu_enabled(false);
            result = tensor * 3.0;
            
            // Check final state
            bool final_state = torch::autocast::is_cpu_enabled();
        }
        
        // Reset autocast state to avoid affecting other tests
        torch::autocast::set_cpu_enabled(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}