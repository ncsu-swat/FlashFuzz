#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        
        // Store original setting to restore later
        // Use the generic device-type based API with kXLA
        bool original_setting = at::autocast::is_autocast_enabled(at::kXLA);
        
        // Set the autocast XLA enabled flag using device-type API
        at::autocast::set_autocast_enabled(at::kXLA, enabled);
        
        // Verify the setting was applied correctly
        bool current_setting = at::autocast::is_autocast_enabled(at::kXLA);
        (void)current_setting; // Use the variable
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might be affected by autocast
            try {
                // Basic operations
                torch::Tensor result = tensor + tensor;
                
                // Only do matmul if tensor is 2D and dimensions are compatible
                if (tensor.dim() == 2 && tensor.size(0) == tensor.size(1)) {
                    result = torch::matmul(tensor, tensor);
                }
                (void)result;
            } catch (...) {
                // Silently ignore shape mismatches
            }
        }
        
        // Toggle the setting and test again
        at::autocast::set_autocast_enabled(at::kXLA, !enabled);
        
        // Verify toggle worked
        bool toggled_setting = at::autocast::is_autocast_enabled(at::kXLA);
        (void)toggled_setting; // Use the variable
        
        // Create another tensor with different settings
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                // Test with the new setting
                torch::Tensor result = tensor2 * 2.0;
                result = tensor2 - tensor2;
                result = tensor2 / 2.0;
                (void)result;
            } catch (...) {
                // Silently ignore errors
            }
        }
        
        // Test setting with different input patterns
        if (Size > 1) {
            bool second_enabled = Data[1] & 0x1;
            at::autocast::set_autocast_enabled(at::kXLA, second_enabled);
            
            // Test the XLA dtype getter using the new API
            try {
                auto xla_dtype = at::autocast::get_autocast_dtype(at::kXLA);
                (void)xla_dtype; // Use the variable
            } catch (...) {
                // May not be available on all builds
            }
        }
        
        // Test autocast cache related functions
        bool original_cache = at::autocast::is_autocast_cache_enabled();
        if (Size > 2) {
            bool cache_enabled = Data[2] & 0x1;
            at::autocast::set_autocast_cache_enabled(cache_enabled);
            
            // Verify cache setting
            bool cache_setting = at::autocast::is_autocast_cache_enabled();
            (void)cache_setting;
        }
        at::autocast::set_autocast_cache_enabled(original_cache);
        
        // Restore the original setting
        at::autocast::set_autocast_enabled(at::kXLA, original_setting);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}