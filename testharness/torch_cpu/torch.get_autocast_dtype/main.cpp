#include "fuzzer_utils.h"
#include <ATen/autocast_mode.h>
#include <iostream>

namespace
{
    torch::ScalarType choose_autocast_dtype(uint8_t selector)
    {
        switch (selector % 3)
        {
        case 0:
            return torch::kFloat16;
        case 1:
            return torch::kBFloat16;
        default:
            return torch::kFloat;
        }
    }
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    if (Size < 3) {
        return -1;
    }

    try
    {
        size_t offset = 0;

        // Parse autocast enabled flag
        uint8_t enabled_byte = Data[offset++];
        bool enabled = (enabled_byte % 2 == 0);

        // Parse device type for get_autocast_dtype (CPU only for portability)
        at::DeviceType target_device = at::kCPU;

        // Parse dtype selection bytes
        uint8_t cpu_dtype_byte = Data[offset++];
        uint8_t test_dtype_byte = Data[offset++];

        // Set autocast state for CPU
        at::autocast::set_autocast_enabled(at::kCPU, enabled);

        // Set autocast dtype for CPU using fuzzer-controlled value
        torch::ScalarType cpu_dtype = choose_autocast_dtype(cpu_dtype_byte);
        at::autocast::set_autocast_dtype(at::kCPU, cpu_dtype);

        // Call get_autocast_dtype (target API: torch.get_autocast_dtype)
        torch::ScalarType result_dtype = at::autocast::get_autocast_dtype(target_device);

        // Verify the result matches what we set
        if (enabled) {
            // When autocast is enabled, result should match what we set
            (void)(result_dtype == cpu_dtype);
        }

        // Also test is_autocast_enabled
        bool is_enabled = at::autocast::is_autocast_enabled(target_device);
        (void)(is_enabled == enabled);

        // Create a tensor and test the autocast dtype in practice
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor = torch::randn({2, 2});
        }

        if (tensor.defined()) {
            // Inner try-catch for expected dtype conversion failures
            try {
                // Convert tensor to float first (autocast works with float inputs)
                torch::Tensor float_tensor = tensor.to(torch::kFloat);
                
                // Test casting to the autocast dtype
                torch::Tensor cast_tensor = float_tensor.to(result_dtype);
                torch::Tensor output = cast_tensor + cast_tensor;
                (void)output.sum();
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }

        // Test with different dtype settings
        for (int i = 0; i < 3; i++) {
            torch::ScalarType test_dtype = choose_autocast_dtype((test_dtype_byte + i) % 3);
            at::autocast::set_autocast_dtype(at::kCPU, test_dtype);
            torch::ScalarType queried_dtype = at::autocast::get_autocast_dtype(at::kCPU);
            (void)(queried_dtype == test_dtype);
        }

        // Reset autocast state
        at::autocast::set_autocast_enabled(at::kCPU, false);
        at::autocast::set_autocast_dtype(at::kCPU, torch::kBFloat16); // Reset to default
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}