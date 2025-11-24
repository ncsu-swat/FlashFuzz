#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        // Keep target op keyword: torch.is_autocast_cpu_enabled
        bool is_enabled = at::autocast::is_autocast_enabled(at::kCPU);
        (void)is_enabled;

        // Try toggling autocast state
        if (Size > 0) {
            bool enable_state = Data[offset] % 2 == 0;
            offset++;

            // Set autocast state
            at::autocast::set_autocast_enabled(at::kCPU, enable_state);

            // Verify the state was set correctly
            bool new_state = at::autocast::is_autocast_enabled(at::kCPU);
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
            at::autocast::set_autocast_enabled(at::kCPU, true);
            result = tensor * 2.0;

            // Disable autocast
            at::autocast::set_autocast_enabled(at::kCPU, false);
            result = tensor * 3.0 + tensor.sum();
            (void)result;

            // Check final state
            bool final_state = at::autocast::is_autocast_enabled(at::kCPU);
            (void)final_state;
        }

        // Reset autocast state to avoid affecting other tests
        at::autocast::set_autocast_enabled(at::kCPU, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
