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

        if (Size < 2) {
            return 0;
        }

        const std::array<c10::DeviceType, 2> device_choices = {
            c10::DeviceType::CPU,
            c10::DeviceType::CUDA,
        };

        // Pick a device from fuzzer data
        uint8_t device_selector = Data[offset++];
        c10::DeviceType device_type = device_choices[device_selector % device_choices.size()];

        // Fall back to CPU if CUDA not available
        if (device_type == c10::DeviceType::CUDA && !torch::cuda::is_available())
        {
            device_type = c10::DeviceType::CPU;
        }

        // Extract enabled state from fuzzer data
        bool enabled = Data[offset++] & 0x1;

        // Get initial state
        bool initial_state = at::autocast::is_autocast_enabled(device_type);
        (void)initial_state;

        // Set autocast enabled state - this is the main API under test
        at::autocast::set_autocast_enabled(device_type, enabled);

        // Verify the state was set
        bool current_state = at::autocast::is_autocast_enabled(device_type);
        (void)current_state;

        // Create a tensor and perform operations that might be affected by autocast
        if (offset < Size)
        {
            try
            {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

                // Perform operations that might be affected by autocast
                torch::Tensor result = tensor + tensor;
                (void)result;

                // Try matrix multiplication if tensor has at least 2 dimensions
                if (tensor.dim() >= 2 && tensor.size(-1) == tensor.size(-2))
                {
                    try
                    {
                        torch::Tensor matmul_result = torch::matmul(tensor, tensor);
                        (void)matmul_result;
                    }
                    catch (...)
                    {
                        // Ignore exceptions from matmul (e.g., shape mismatch)
                    }
                }

                // Try some other operations that might be affected by autocast
                torch::Tensor sin_result = torch::sin(tensor);
                torch::Tensor exp_result = torch::exp(tensor);
                (void)sin_result;
                (void)exp_result;

                // Linear operations are common autocast targets
                if (tensor.dim() >= 2)
                {
                    try
                    {
                        auto linear_result = torch::nn::functional::linear(
                            tensor, 
                            torch::randn({tensor.size(-1), tensor.size(-1)})
                        );
                        (void)linear_result;
                    }
                    catch (...)
                    {
                        // Ignore shape/type errors
                    }
                }
            }
            catch (...)
            {
                // Ignore tensor creation/operation failures
            }
        }

        // Toggle state multiple times based on fuzzer data
        while (offset < Size)
        {
            bool toggle_state = Data[offset++] & 0x1;
            at::autocast::set_autocast_enabled(device_type, toggle_state);
            bool check_state = at::autocast::is_autocast_enabled(device_type);
            (void)check_state;
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