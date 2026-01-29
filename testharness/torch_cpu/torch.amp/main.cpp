#include "fuzzer_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor from fuzz data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor safe_tensor = input_tensor.flatten();
        if (safe_tensor.numel() < 4)
        {
            safe_tensor = safe_tensor.numel() > 0 
                ? torch::cat({safe_tensor, torch::zeros({4 - safe_tensor.numel()})})
                : torch::zeros({4});
        }
        
        int64_t side = std::min<int64_t>(8, static_cast<int64_t>(std::sqrt(static_cast<double>(safe_tensor.numel()))));
        side = std::max<int64_t>(side, 2);
        safe_tensor = safe_tensor.narrow(0, 0, std::min<int64_t>(safe_tensor.numel(), side * side)).reshape({side, side});
        safe_tensor = safe_tensor.to(torch::kFloat);

        // Extract fuzzer-controlled parameters
        bool enabled = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        c10::DeviceType device_type = c10::kCPU;
        
        at::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            switch (dtype_selector % 3) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kBFloat16; break;
                default: dtype = torch::kFloat; break;  // kHalf may not be well supported on CPU
            }
        }
        
        bool cache_enabled = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Save original autocast state
        bool orig_enabled = at::autocast::is_autocast_enabled(device_type);
        at::ScalarType orig_dtype = at::autocast::get_autocast_dtype(device_type);
        bool orig_cache = at::autocast::is_autocast_cache_enabled();
        int orig_nesting = 0;

        try {
            // Test autocast enable/disable
            at::autocast::set_autocast_enabled(device_type, enabled);
            at::autocast::set_autocast_dtype(device_type, dtype);
            at::autocast::set_autocast_cache_enabled(cache_enabled);
            
            // Verify state was set
            bool is_enabled = at::autocast::is_autocast_enabled(device_type);
            at::ScalarType current_dtype = at::autocast::get_autocast_dtype(device_type);
            bool is_cache_enabled = at::autocast::is_autocast_cache_enabled();
            (void)is_enabled;
            (void)current_dtype;
            (void)is_cache_enabled;
            
            // Test nesting increment/decrement
            at::autocast::increment_nesting();
            orig_nesting = 1;
            
            // Perform operations under autocast context
            torch::Tensor result1 = torch::matmul(safe_tensor, safe_tensor);
            (void)result1.sum().item<float>();
            
            // Test with another dtype
            at::autocast::set_autocast_dtype(device_type, torch::kFloat);
            torch::Tensor result2 = torch::mm(safe_tensor, safe_tensor);
            (void)result2.sum().item<float>();
            
            // Test clear cache
            at::autocast::clear_cache();
            
            // Test toggle enabled state
            at::autocast::set_autocast_enabled(device_type, !enabled);
            at::autocast::set_autocast_enabled(device_type, enabled);
            
            // Decrement nesting
            at::autocast::decrement_nesting();
            orig_nesting = 0;
            
            // Test nested autocast
            at::autocast::increment_nesting();
            orig_nesting = 1;
            at::autocast::increment_nesting();
            orig_nesting = 2;
            
            torch::Tensor result3 = torch::addmm(safe_tensor, safe_tensor, safe_tensor);
            (void)result3.sum().item<float>();
            
            at::autocast::decrement_nesting();
            orig_nesting = 1;
            at::autocast::decrement_nesting();
            orig_nesting = 0;
        }
        catch (...) {
            // Clean up nesting on error
            while (orig_nesting > 0) {
                at::autocast::decrement_nesting();
                orig_nesting--;
            }
        }

        // Always restore original state
        at::autocast::set_autocast_cache_enabled(orig_cache);
        at::autocast::set_autocast_dtype(device_type, orig_dtype);
        at::autocast::set_autocast_enabled(device_type, orig_enabled);

        // Test basic tensor operations that AMP typically affects
        // Use fuzzer-derived tensor for determinism
        {
            int64_t batch = 1;
            int64_t in_channels = 3;
            int64_t height = 8;
            int64_t width = 8;
            
            // Create conv input from fuzz data
            torch::Tensor conv_input = fuzzer_utils::createTensor(Data, Size, offset);
            if (conv_input.numel() < batch * in_channels * height * width) {
                conv_input = torch::zeros({batch, in_channels, height, width});
            } else {
                conv_input = conv_input.flatten()
                    .narrow(0, 0, batch * in_channels * height * width)
                    .reshape({batch, in_channels, height, width})
                    .to(torch::kFloat);
            }
            
            // Create weight from fuzz data
            torch::Tensor conv_weight = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t out_channels = 4;
            int64_t kernel = 3;
            if (conv_weight.numel() < out_channels * in_channels * kernel * kernel) {
                conv_weight = torch::zeros({out_channels, in_channels, kernel, kernel});
            } else {
                conv_weight = conv_weight.flatten()
                    .narrow(0, 0, out_channels * in_channels * kernel * kernel)
                    .reshape({out_channels, in_channels, kernel, kernel})
                    .to(torch::kFloat);
            }
            
            try {
                torch::Tensor conv_output = torch::conv2d(conv_input, conv_weight);
                (void)conv_output.sum().item<float>();
            }
            catch (...) {
                // Shape mismatches are expected with fuzz data
            }
        }

        // Test GradScaler-related operations (manual scaling simulation)
        {
            torch::Tensor scale_factor = torch::tensor({1.0f});
            if (offset < Size) {
                float scale_val = static_cast<float>(Data[offset++]) / 10.0f + 0.1f;
                scale_factor = torch::tensor({scale_val});
            }
            
            torch::Tensor scaled = safe_tensor * scale_factor;
            torch::Tensor unscaled = scaled / scale_factor;
            (void)unscaled.sum().item<float>();
            
            // Check for inf/nan (common in AMP workflows)
            bool has_inf = torch::isinf(scaled).any().item<bool>();
            bool has_nan = torch::isnan(scaled).any().item<bool>();
            (void)has_inf;
            (void)has_nan;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}