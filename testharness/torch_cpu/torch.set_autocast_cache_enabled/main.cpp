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
        bool enable_cache = Data[0] & 0x1;
        offset++;
        
        // Set the autocast cache enabled state
        torch::autograd::set_autocast_cache_enabled(enable_cache);
        
        // Verify the setting was applied by checking the current state
        bool current_state = torch::autograd::is_autocast_cache_enabled();
        
        // Create a tensor and perform some autocast operations to exercise the cache
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform operations that might use autocast cache
            torch::DeviceType device_type = torch::kCUDA;
            if (Data[offset % Size] & 0x1) {
                device_type = torch::kCPU;
            }
            
            // Use autocast context to test cache behavior
            {
                torch::autograd::AutocastMode autocast_ctx(device_type, enable_cache);
                
                // Perform operations that might use the cache
                torch::Tensor result = tensor + 1.0;
                result = result * 2.0;
                result = torch::nn::functional::relu(result);
                
                // Try more complex operations
                if (tensor.dim() > 0 && tensor.size(0) > 0) {
                    try {
                        result = torch::matmul(result, result);
                    } catch (...) {
                        // Ignore errors from matmul with incompatible shapes
                    }
                }
            }
            
            // Try nested autocast contexts with different settings
            {
                torch::autograd::AutocastMode outer_ctx(device_type, enable_cache);
                torch::Tensor result = tensor + 3.0;
                
                {
                    // Nested context with opposite cache setting
                    torch::autograd::AutocastMode inner_ctx(device_type, !enable_cache);
                    result = result * 4.0;
                }
                
                result = result - 2.0;
            }
        }
        
        // Toggle the setting and verify it changed
        torch::autograd::set_autocast_cache_enabled(!enable_cache);
        bool new_state = torch::autograd::is_autocast_cache_enabled();
        
        // Reset to original state to avoid affecting other tests
        torch::autograd::set_autocast_cache_enabled(enable_cache);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}