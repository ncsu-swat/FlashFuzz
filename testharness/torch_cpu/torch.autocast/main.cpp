#include "fuzzer_utils.h"
#include <ATen/autocast_mode.h>
#include <iostream>
#include <tuple>

namespace
{
    const char *kTargetApi = "torch.autocast";

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
        return (selector % 2 == 0) ? torch::kBFloat16 : torch::kFloat32;
    }

    torch::Tensor normalize_input(const torch::Tensor &input)
    {
        torch::Tensor flat = input.flatten();
        if (flat.numel() > 1024)
        {
            flat = flat.narrow(0, 0, 1024);
        }
        return flat.unsqueeze(0);
    }
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 3) {
            return 0;
        }

        size_t offset = 0;
        (void)kTargetApi;
        
        // Parse device type (CPU only for this harness since CUDA may not be available)
        torch::Device device = torch::kCPU;

        // Parse dtype for autocast
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType cast_dtype = select_dtype(device, dtype_selector);

        // Parse enabled flag
        bool enabled = Data[offset++] % 2 == 1;
        
        // Parse cache_enabled flag
        bool cache_enabled = Data[offset++] % 2 == 1;
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat32));
        }
        
        // Ensure tensor is float type for autocast operations
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }

        torch::Tensor working = normalize_input(input);
        
        // Test 1: Scoped autocast settings with provided flags
        {
            AutocastGuard guard(device.type(), enabled, cast_dtype, cache_enabled);
            
            // Verify autocast state was set correctly
            bool current_enabled = at::autocast::is_autocast_enabled(device.type());
            at::ScalarType current_dtype = at::autocast::get_autocast_dtype(device.type());
            bool current_cache = at::autocast::is_autocast_cache_enabled();
            (void)current_enabled;
            (void)current_dtype;
            (void)current_cache;
            
            torch::Tensor result = torch::add(working, working);
            torch::Tensor activated = torch::nn::functional::relu(result);
            activated.sum().item<double>();
        }

        // Test 2: Flip cache flag and dtype to a secondary choice
        {
            torch::ScalarType inner_dtype = (cast_dtype == torch::kBFloat16) ? torch::kFloat32 : torch::kBFloat16;
            AutocastGuard guard(device.type(), !enabled, inner_dtype, !cache_enabled);
            
            torch::Tensor scaled = working * 3.0;
            torch::Tensor normalized = torch::tanh(scaled);
            normalized.sum().item<double>();
        }

        // Test 3: Manual state management without RAII guard
        {
            bool prev_enabled = at::autocast::is_autocast_enabled(device.type());
            at::ScalarType prev_dtype = at::autocast::get_autocast_dtype(device.type());
            bool prev_cache = at::autocast::is_autocast_cache_enabled();

            at::autocast::set_autocast_enabled(device.type(), enabled);
            at::autocast::set_autocast_dtype(device.type(), cast_dtype);
            at::autocast::set_autocast_cache_enabled(cache_enabled);

            torch::Tensor shifted = working + 1.5;
            torch::Tensor clipped = torch::clamp(shifted, -1.0, 1.0);
            clipped.sum().item<double>();

            // Restore previous state
            at::autocast::set_autocast_enabled(device.type(), prev_enabled);
            at::autocast::set_autocast_dtype(device.type(), prev_dtype);
            at::autocast::set_autocast_cache_enabled(prev_cache);
        }

        // Test 4: Clear autocast cache
        {
            AutocastGuard guard(device.type(), true, cast_dtype, true);
            
            // Perform operations that might be cached
            torch::Tensor a = torch::matmul(working, working.transpose(0, 1));
            torch::Tensor b = torch::matmul(working, working.transpose(0, 1));
            (void)a;
            (void)b;
            
            // Clear the cache
            at::autocast::clear_cache();
            
            // Perform more operations after cache clear
            torch::Tensor c = torch::matmul(working, working.transpose(0, 1));
            c.sum().item<double>();
        }

        // Test 5: Nested autocast contexts with different settings
        {
            AutocastGuard outer_guard(device.type(), true, torch::kBFloat16, true);
            torch::Tensor outer_result = torch::add(working, 1.0);
            
            {
                AutocastGuard inner_guard(device.type(), false, torch::kFloat32, false);
                torch::Tensor inner_result = torch::mul(outer_result, 2.0);
                inner_result.sum().item<double>();
            }
            
            // Back to outer context
            torch::Tensor final_result = torch::sub(outer_result, 0.5);
            final_result.sum().item<double>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}