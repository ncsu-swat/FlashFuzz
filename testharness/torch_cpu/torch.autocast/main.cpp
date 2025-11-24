#include "fuzzer_utils.h"    // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream>          // For cerr
#include <tuple>             // For std::get with lu_unpack result

namespace
{
    // Target API keyword retained for harness checks.
    const char *kTargetApi = "torch.autocast";

    // RAII guard to flip autocast settings and restore on scope exit.
    struct AutocastGuard
    {
        AutocastGuard(c10::DeviceType device_type, bool enabled, at::ScalarType dtype, bool cache_enabled)
            : device_type_(device_type)
        {
            prev_enabled_ = at::autocast::is_autocast_enabled(device_type_);
            prev_dtype_ = at::autocast::get_autocast_dtype(device_type_);
            prev_cache_enabled_ = at::autocast::is_autocast_cache_enabled();

            at::autocast::set_autocast_enabled(device_type_, enabled);
            at::autocast::set_autocast_dtype(device_type_, dtype);
            at::autocast::set_autocast_cache_enabled(cache_enabled);
        }

        ~AutocastGuard()
        {
            at::autocast::set_autocast_enabled(device_type_, prev_enabled_);
            at::autocast::set_autocast_dtype(device_type_, prev_dtype_);
            at::autocast::set_autocast_cache_enabled(prev_cache_enabled_);
        }

    private:
        c10::DeviceType device_type_;
        bool prev_enabled_{false};
        at::ScalarType prev_dtype_{at::kFloat};
        bool prev_cache_enabled_{true};
    };

    at::ScalarType select_dtype(const torch::Device &device, uint8_t selector)
    {
        if (device.is_cuda())
        {
            switch (selector % 3)
            {
            case 0:
                return torch::kFloat16;
            case 1:
                return torch::kBFloat16;
            default:
                return torch::kFloat32;
            }
        }

        // CPU and other backends commonly use bf16 or fp32 for autocast.
        return (selector % 2 == 0) ? torch::kBFloat16 : torch::kFloat32;
    }

    torch::Tensor normalize_input(const torch::Tensor &input)
    {
        torch::Tensor flat = input.flatten();
        if (flat.numel() > 1024)
        {
            flat = flat.narrow(0, 0, 1024);
        }
        // Keep a consistent leading dimension for later ops.
        return flat.unsqueeze(0);
    }
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }

        (void)kTargetApi;
        
        // Parse device type (CPU or CUDA)
        bool use_cuda = Data[offset++] % 2 == 1 && torch::cuda::is_available();
        torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;

        // Parse dtype for autocast
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType cast_dtype = select_dtype(device, dtype_selector);

        // Parse enabled flag
        bool enabled = Data[offset++] % 2 == 1;
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            input = input.to(device);
        } else {
            // Create a default tensor if we don't have enough data
            input = torch::rand({2, 3}, torch::TensorOptions().device(device));
        }
        
        // Parse cache_enabled flag if we have more data
        bool cache_enabled = true;
        if (offset < Size) {
            cache_enabled = Data[offset++] % 2 == 1;
        }

        torch::Tensor working = normalize_input(input);
        
        // Test autocast in different ways

        // 1. Scoped autocast settings with provided flags.
        {
            AutocastGuard guard(device.type(), enabled, cast_dtype, cache_enabled);
            torch::Tensor result = torch::add(working, working);
            torch::Tensor activated = torch::nn::functional::relu(result);
            activated.sum().item<double>();
        }

        // 2. Flip cache flag and dtype to a secondary choice.
        {
            torch::ScalarType inner_dtype = (cast_dtype == torch::kBFloat16) ? torch::kFloat32 : torch::kBFloat16;
            AutocastGuard guard(device.type(), !enabled, inner_dtype, !cache_enabled);
            torch::Tensor scaled = working * 3.0;
            torch::Tensor normalized = torch::tanh(scaled);
            normalized.sum().item<double>();
        }

        // 3. Query the current autocast state and adjust once.
        {
            bool prev_enabled = at::autocast::is_autocast_enabled(device.type());
            at::ScalarType prev_dtype = at::autocast::get_autocast_dtype(device.type());
            bool prev_cache = at::autocast::is_autocast_cache_enabled();

            at::autocast::set_autocast_enabled(device.type(), enabled);
            at::autocast::set_autocast_dtype(device.type(), cast_dtype);
            at::autocast::set_autocast_cache_enabled(cache_enabled);

            bool is_enabled = at::autocast::is_autocast_enabled(device.type());
            (void)is_enabled;
            torch::Tensor shifted = working + 1.5;
            torch::Tensor clipped = torch::clamp(shifted, -1.0, 1.0);
            clipped.sum().item<double>();

            at::autocast::set_autocast_enabled(device.type(), prev_enabled);
            at::autocast::set_autocast_dtype(device.type(), prev_dtype);
            at::autocast::set_autocast_cache_enabled(prev_cache);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
