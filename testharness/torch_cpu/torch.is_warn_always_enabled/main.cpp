#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if warning system is enabled
        bool is_warn_enabled = torch::is_warn_always_enabled();
        
        // Try with different tensor types to see if they affect warning behavior
        if (Size > 0) {
            // Create a tensor to potentially trigger warnings
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Check warning status again after tensor creation
            bool is_warn_enabled_after = torch::is_warn_always_enabled();
            
            // Try operations that might trigger warnings
            if (tensor.defined()) {
                // Try to perform operations that might trigger warnings
                try {
                    // Attempt division by zero which might trigger warnings
                    if (tensor.numel() > 0) {
                        torch::Tensor zeros = torch::zeros_like(tensor);
                        torch::Tensor result = tensor / zeros;
                    }
                } catch (...) {
                    // Ignore exceptions from division by zero
                }
                
                // Check if warning status changed
                bool is_warn_enabled_final = torch::is_warn_always_enabled();
            }
        }
        
        // Test the function with different warning configurations
        // This doesn't actually change the warning state but exercises the API
        for (int i = 0; i < 2 && offset < Size; i++) {
            uint8_t byte = Data[offset++];
            bool should_warn = byte % 2 == 0;
            
            // Check warning status
            bool current_status = torch::is_warn_always_enabled();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}