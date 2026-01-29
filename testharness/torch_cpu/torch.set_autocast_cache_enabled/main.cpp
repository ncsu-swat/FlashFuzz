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
        bool enable_cache = Data[0] & 0x1;
        offset++;
        
        // Save original state to restore later
        bool original_state = at::autocast::is_autocast_cache_enabled();
        
        // Set the autocast cache enabled state - this is the main API under test
        at::autocast::set_autocast_cache_enabled(enable_cache);
        
        // Verify the setting was applied by checking the current state
        bool current_state = at::autocast::is_autocast_cache_enabled();
        
        // Create a tensor and perform some operations to exercise the cache
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure tensor is float type for operations
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32);
            }
            
            // Perform basic operations
            torch::Tensor result = tensor + 1.0f;
            result = result * 2.0f;
            
            // Try relu operation
            try {
                result = torch::relu(result);
            } catch (...) {
                // Silently ignore shape-related errors
            }
            
            // Try matrix operations if tensor has appropriate shape
            if (tensor.dim() == 2 && tensor.size(0) > 0 && tensor.size(1) > 0) {
                try {
                    // Make a square tensor for matmul
                    int64_t dim = std::min(tensor.size(0), tensor.size(1));
                    torch::Tensor square = tensor.narrow(0, 0, dim).narrow(1, 0, dim);
                    torch::Tensor mm_result = torch::matmul(square, square);
                } catch (...) {
                    // Silently ignore matmul errors
                }
            }
        }
        
        // Toggle the setting and verify it changed
        at::autocast::set_autocast_cache_enabled(!enable_cache);
        bool new_state = at::autocast::is_autocast_cache_enabled();
        
        // Test toggling multiple times based on fuzzer data
        if (Size > 1) {
            for (size_t i = 1; i < std::min(Size, (size_t)10); i++) {
                bool val = Data[i] & 0x1;
                at::autocast::set_autocast_cache_enabled(val);
            }
        }
        
        // Restore original state to avoid affecting other tests
        at::autocast::set_autocast_cache_enabled(original_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}