#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the mode selection
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine the debug mode
        uint8_t mode_byte = Data[offset++];
        
        // Map the byte to one of the three possible modes:
        // 0 = OFF, 1 = WARN, 2 = ERROR
        int mode;
        if (mode_byte % 3 == 0) {
            mode = 0; // OFF
        } else if (mode_byte % 3 == 1) {
            mode = 1; // WARN
        } else {
            mode = 2; // ERROR
        }
        
        // Set the deterministic debug mode
        torch::set_deterministic_debug_mode(static_cast<torch::DeterministicDebugMode>(mode));
        
        // Create a tensor to test if the mode affects operations
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might be affected by deterministic mode
            if (tensor.dim() > 0) {
                auto result = torch::max_pool2d(tensor, {2, 2});
            }
            
            // Test with another operation
            if (tensor.numel() > 0) {
                auto result = torch::conv2d(tensor, tensor, {});
            }
            
            // Reset to default mode (OFF) before exiting
            torch::set_deterministic_debug_mode(torch::DeterministicDebugMode::Off);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}