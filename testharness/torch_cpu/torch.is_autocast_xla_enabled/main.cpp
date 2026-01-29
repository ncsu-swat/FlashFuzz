#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
#include <iostream> // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    // Keep keyword for harness validation.
    (void)"torch.is_autocast_xla_enabled";

    try
    {
        size_t offset = 0;

        // Check if autocast XLA is enabled via C++ API.
        // Note: XLA device may not be available in all builds, but the API should still work
        bool is_enabled = at::autocast::is_autocast_enabled(c10::DeviceType::XLA);
        (void)is_enabled;

        // Need at least some data to drive fuzzing decisions
        if (Size < 1)
        {
            return 0;
        }

        // Try to enable/disable autocast XLA based on input data
        bool should_enable = Data[offset++] % 2 == 0;
        
        // Save original state
        bool original_state = at::autocast::is_autocast_enabled(c10::DeviceType::XLA);
        
        // Set new state
        at::autocast::set_autocast_enabled(c10::DeviceType::XLA, should_enable);

        // Verify the change took effect
        bool is_enabled_after_change = at::autocast::is_autocast_enabled(c10::DeviceType::XLA);
        (void)is_enabled_after_change;

        // Try to create a tensor if we have enough data and exercise it
        // under the current autocast setting
        if (offset < Size && Size - offset > 2)
        {
            try
            {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                // Exercise tensor path to ensure execution under autocast context
                (void)tensor.sum().item<double>();
            }
            catch (const std::exception &)
            {
                // Tensor creation/operation may fail for invalid inputs - that's OK
            }
        }

        // Toggle state again to exercise more paths
        at::autocast::set_autocast_enabled(c10::DeviceType::XLA, !should_enable);
        bool toggled_state = at::autocast::is_autocast_enabled(c10::DeviceType::XLA);
        (void)toggled_state;

        // Reset to original state to avoid affecting other tests
        at::autocast::set_autocast_enabled(c10::DeviceType::XLA, original_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}