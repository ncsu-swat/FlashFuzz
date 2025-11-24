#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
#include <iostream> // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Keep keyword for harness validation.
    (void)"torch.is_autocast_xla_enabled";

    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        // Check if autocast XLA is enabled via C++ API.
        bool is_enabled = at::autocast::is_autocast_enabled(c10::DeviceType::XLA);

        // Try to create a tensor if we have enough data
        if (Size > 2)
        {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Exercise tensor path to ensure execution.
            (void)tensor.sum().item<double>();

            // Check again after tensor creation to see if there's any change
            bool is_enabled_after = at::autocast::is_autocast_enabled(c10::DeviceType::XLA);
            (void)is_enabled_after;

            // Try to enable/disable autocast XLA based on input data
            if (offset < Size)
            {
                bool should_enable = Data[offset++] % 2 == 0;
                at::autocast::set_autocast_enabled(c10::DeviceType::XLA, should_enable);

                // Verify the change took effect
                bool is_enabled_after_change = at::autocast::is_autocast_enabled(c10::DeviceType::XLA);
                (void)is_enabled_after_change;

                // Reset to original state
                at::autocast::set_autocast_enabled(c10::DeviceType::XLA, is_enabled);
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
