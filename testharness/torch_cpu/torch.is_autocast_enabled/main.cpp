#include "fuzzer_utils.h"
#include <ATen/autocast_mode.h>
#include <array>
#include <iostream>

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

        const std::array<c10::DeviceType, 2> device_choices = {
            c10::DeviceType::CPU,
            c10::DeviceType::CUDA,
        };

        // Test default behavior with CPU (always available)
        bool cpu_enabled_initial = at::autocast::is_autocast_enabled(c10::DeviceType::CPU);
        (void)cpu_enabled_initial;

        // Test with CUDA if available
        if (torch::cuda::is_available())
        {
            bool cuda_enabled_initial = at::autocast::is_autocast_enabled(c10::DeviceType::CUDA);
            (void)cuda_enabled_initial;
        }

        if (Size > 0)
        {
            // Pick a device from fuzzer data
            uint8_t selector = Data[offset % Size];
            offset++;
            c10::DeviceType device_type = device_choices[selector % device_choices.size()];
            
            // Fall back to CPU if CUDA not available
            if (device_type == c10::DeviceType::CUDA && !torch::cuda::is_available())
            {
                device_type = c10::DeviceType::CPU;
            }

            // Test the full cycle of autocast state
            bool before_toggle = at::autocast::is_autocast_enabled(device_type);

            // Enable autocast
            at::autocast::set_autocast_enabled(device_type, true);
            bool after_enable = at::autocast::is_autocast_enabled(device_type);

            // Create a tensor and do a simple operation under autocast
            try
            {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = (tensor + 1).sum();
                (void)result;
            }
            catch (...)
            {
                // Ignore tensor creation/operation failures
            }

            // Disable autocast
            at::autocast::set_autocast_enabled(device_type, false);
            bool after_disable = at::autocast::is_autocast_enabled(device_type);

            // Use values to prevent optimization
            volatile bool b1 = before_toggle;
            volatile bool b2 = after_enable;
            volatile bool b3 = after_disable;
            (void)b1;
            (void)b2;
            (void)b3;

            // Test toggling based on fuzzer input
            if (Size > offset)
            {
                bool enable_state = (Data[offset % Size] % 2) == 0;
                offset++;
                at::autocast::set_autocast_enabled(device_type, enable_state);
                bool current_state = at::autocast::is_autocast_enabled(device_type);
                (void)current_state;
            }
        }

        // Reset to disabled state to avoid leaking state across runs
        at::autocast::set_autocast_enabled(c10::DeviceType::CPU, false);
        if (torch::cuda::is_available())
        {
            at::autocast::set_autocast_enabled(c10::DeviceType::CUDA, false);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}