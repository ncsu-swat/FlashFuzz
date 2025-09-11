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
        
        // Check if autocast XLA is enabled
        bool is_enabled = torch::is_autocast_xla_enabled();
        
        // Try to create a tensor if we have enough data
        if (Size > 2) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Check again after tensor creation to see if there's any change
            bool is_enabled_after = torch::is_autocast_xla_enabled();
            
            // Try to enable/disable autocast XLA based on input data
            if (offset < Size) {
                bool should_enable = Data[offset++] % 2 == 0;
                torch::set_autocast_xla_enabled(should_enable);
                
                // Verify the change took effect
                bool is_enabled_after_change = torch::is_autocast_xla_enabled();
                
                // Reset to original state
                torch::set_autocast_xla_enabled(is_enabled);
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
