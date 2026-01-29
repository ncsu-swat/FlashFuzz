#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Get original state to restore later
        bool original_state = at::autocast::is_cpu_enabled();
        
        // Set autocast CPU enabled state - this is the main API under test
        at::autocast::set_cpu_enabled(enabled);
        
        // Verify the state was set correctly
        bool current_state = at::autocast::is_cpu_enabled();
        if (current_state != enabled) {
            std::cerr << "State mismatch after set" << std::endl;
        }
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure tensor is float type for autocast to be meaningful
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32);
            }
            
            // Perform operations that might be affected by autocast
            // These operations should work regardless of autocast state
            try {
                auto result1 = tensor + tensor;
                auto result2 = tensor * tensor;
                
                // Force computation
                result1.sum();
                result2.sum();
            } catch (const std::exception &) {
                // Silently ignore expected failures (e.g., shape issues)
            }
            
            // Try matmul which requires compatible shapes
            try {
                if (tensor.dim() >= 2) {
                    auto t = tensor.view({tensor.size(0), -1});
                    auto t_t = t.t();
                    auto matmul_result = torch::matmul(t, t_t);
                    matmul_result.sum();
                }
            } catch (const std::exception &) {
                // Silently ignore shape incompatibilities
            }
        }
        
        // Toggle the autocast state and test again
        at::autocast::set_cpu_enabled(!enabled);
        
        // Verify toggle worked
        current_state = at::autocast::is_cpu_enabled();
        if (current_state != !enabled) {
            std::cerr << "State mismatch after toggle" << std::endl;
        }
        
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32);
            }
            
            try {
                auto result = tensor + tensor;
                result.sum();
            } catch (const std::exception &) {
                // Silently ignore
            }
        }
        
        // Test setting to explicit values
        at::autocast::set_cpu_enabled(true);
        at::autocast::set_cpu_enabled(false);
        
        // Restore original state
        at::autocast::set_cpu_enabled(original_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}