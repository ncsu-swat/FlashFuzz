#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>  // PyTorch C++ frontend

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
        
        // Extract boolean value from fuzzer input
        bool flush_denormal = (Data[offset] % 2) == 1;
        offset++;
        
        // Store original state to restore later
        bool original_state = torch::is_flush_denormal();
        
        // Test setting flush denormal mode
        torch::set_flush_denormal(flush_denormal);
        
        // Verify the setting was applied
        bool current_state = torch::is_flush_denormal();
        if (current_state != flush_denormal) {
            std::cout << "Warning: set_flush_denormal(" << flush_denormal 
                      << ") did not set expected state. Got: " << current_state << std::endl;
        }
        
        // Test multiple rapid toggles if we have more data
        if (Size > 1) {
            for (size_t i = 1; i < Size && i < 100; ++i) {
                bool toggle_state = (Data[i] % 2) == 1;
                torch::set_flush_denormal(toggle_state);
                
                // Verify each toggle
                bool verify_state = torch::is_flush_denormal();
                if (verify_state != toggle_state) {
                    std::cout << "Warning: rapid toggle failed at iteration " << i 
                              << ". Expected: " << toggle_state << ", Got: " << verify_state << std::endl;
                }
            }
        }
        
        // Test edge case: setting the same value multiple times
        torch::set_flush_denormal(true);
        torch::set_flush_denormal(true);
        torch::set_flush_denormal(true);
        
        torch::set_flush_denormal(false);
        torch::set_flush_denormal(false);
        torch::set_flush_denormal(false);
        
        // Test alternating pattern
        for (int i = 0; i < 10; ++i) {
            torch::set_flush_denormal(i % 2 == 0);
        }
        
        // Restore original state to avoid affecting other tests
        torch::set_flush_denormal(original_state);
        
        // Final verification that restore worked
        bool final_state = torch::is_flush_denormal();
        if (final_state != original_state) {
            std::cout << "Warning: failed to restore original flush denormal state" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}