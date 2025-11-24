#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <array>
#include <iostream> // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // torch.is_autocast_enabled
        const std::array<c10::DeviceType, 2> device_choices = {
            c10::DeviceType::CUDA,
            c10::DeviceType::CPU,
        };

        // Respect the Python default of CUDA when available.
        c10::DeviceType default_device = torch::cuda::is_available() ? c10::DeviceType::CUDA
                                                                     : c10::DeviceType::CPU;
        bool default_enabled = at::autocast::is_autocast_enabled(default_device);
        (void)default_enabled;

        if (Size > 0)
        {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

            // Pick a device from fuzzer data and fall back to CPU if unsupported.
            uint8_t selector = Data[offset % Size];
            c10::DeviceType device_type = device_choices[selector % device_choices.size()];
            if (device_type == c10::DeviceType::CUDA && !torch::cuda::is_available())
            {
                device_type = c10::DeviceType::CPU;
            }

            bool before_toggle = at::autocast::is_autocast_enabled(device_type);
            at::autocast::set_autocast_enabled(device_type, true);
            bool after_enable = at::autocast::is_autocast_enabled(device_type);

            // Exercise autocast state while using a small tensor op.
            (tensor + 1).sum();

            at::autocast::set_autocast_enabled(device_type, false);
            bool after_disable = at::autocast::is_autocast_enabled(device_type);

            // Prevent unused variable warnings and keep the values observable.
            if (before_toggle == after_enable && after_disable)
            {
                tensor = tensor.relu();
            }
        }

        // Reset known devices to a disabled state to avoid leaking state across runs.
        at::autocast::set_autocast_enabled(c10::DeviceType::CPU, false);
        if (torch::cuda::is_available())
        {
            at::autocast::set_autocast_enabled(c10::DeviceType::CUDA, false);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
