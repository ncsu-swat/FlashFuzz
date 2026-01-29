#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream> // For cerr
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;

        // Save original autocast state to restore later
        bool original_state = at::autocast::is_autocast_enabled(at::kCPU);

        // Keep target op keyword: torch.is_autocast_cpu_enabled
        // Test the main API - check if autocast is enabled for CPU
        bool is_enabled = at::autocast::is_autocast_enabled(at::kCPU);
        (void)is_enabled;

        // Try toggling autocast state based on fuzzer input
        if (Size > 0) {
            bool enable_state = Data[offset] % 2 == 0;
            offset++;

            // Set autocast state
            at::autocast::set_autocast_enabled(at::kCPU, enable_state);

            // Verify the state was set correctly using the target API
            bool new_state = at::autocast::is_autocast_enabled(at::kCPU);
            if (new_state != enable_state) {
                // This would indicate a bug in PyTorch
                std::cerr << "Autocast state mismatch: expected " << enable_state 
                          << " got " << new_state << std::endl;
            }
        }

        // Create a tensor to test with autocast if we have more data
        if (Size > offset) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

            // Test operations with different autocast states
            try {
                // Enable autocast and perform operations
                at::autocast::set_autocast_enabled(at::kCPU, true);
                
                // Verify state using target API
                bool state_check1 = at::autocast::is_autocast_enabled(at::kCPU);
                (void)state_check1;
                
                torch::Tensor result1 = tensor * 2.0f;
                torch::Tensor result2 = tensor + tensor;
                torch::Tensor result3 = torch::matmul(tensor.view({-1, 1}), tensor.view({1, -1}));
                (void)result1;
                (void)result2;
                (void)result3;

                // Disable autocast and perform operations
                at::autocast::set_autocast_enabled(at::kCPU, false);
                
                // Verify state using target API
                bool state_check2 = at::autocast::is_autocast_enabled(at::kCPU);
                (void)state_check2;
                
                torch::Tensor result4 = tensor * 3.0f + tensor.sum();
                (void)result4;
            }
            catch (const std::exception &) {
                // Shape mismatches or other expected errors - ignore silently
            }
        }

        // Test rapid toggling of autocast state
        if (Size > 1) {
            for (size_t i = 0; i < std::min(Size, (size_t)10); i++) {
                bool toggle = Data[i % Size] % 2 == 0;
                at::autocast::set_autocast_enabled(at::kCPU, toggle);
                
                // Always verify using target API
                bool verified = at::autocast::is_autocast_enabled(at::kCPU);
                (void)verified;
            }
        }

        // Restore original autocast state to avoid affecting other iterations
        at::autocast::set_autocast_enabled(at::kCPU, original_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}