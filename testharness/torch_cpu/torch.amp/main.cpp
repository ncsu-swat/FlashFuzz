#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>      // For std::min/max
#include <cmath>          // For std::sqrt
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <ATen/autocast_mode.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor safe_tensor = input_tensor.flatten();
        if (safe_tensor.numel() < 4)
        {
            safe_tensor = torch::randn({2, 2});
        }
        else
        {
            int64_t side = std::min<int64_t>(8, static_cast<int64_t>(std::sqrt(static_cast<double>(safe_tensor.numel()))));
            side = std::max<int64_t>(side, 2);
            safe_tensor = safe_tensor.narrow(0, 0, std::min<int64_t>(safe_tensor.numel(), side * side)).reshape({side, side});
        }
        
        // Test torch.amp.autocast
        if (offset < Size) {
            bool enabled = Data[offset++] % 2 == 0;
            
            // Get device type
            c10::DeviceType device_type = c10::kCPU;
            
            // Get dtype
            at::ScalarType dtype = torch::kFloat;
            if (offset < Size) {
                uint8_t dtype_selector = Data[offset++];
                if (dtype_selector % 3 == 0) {
                    dtype = torch::kFloat;
                } else if (dtype_selector % 3 == 1) {
                    dtype = torch::kHalf;
                } else {
                    dtype = torch::kBFloat16;
                }
            }
            
            // Get cache enabled flag
            bool cache_enabled = true;
            if (offset < Size) {
                cache_enabled = Data[offset++] % 2 == 0;
            }
            
            // Test autocast using at::autocast APIs directly
            {
                bool prev_enabled = at::autocast::is_autocast_enabled(device_type);
                at::ScalarType prev_dtype = at::autocast::get_autocast_dtype(device_type);
                bool prev_cache = at::autocast::is_autocast_cache_enabled();
                
                at::autocast::set_autocast_enabled(device_type, enabled);
                at::autocast::set_autocast_dtype(device_type, dtype);
                at::autocast::set_autocast_cache_enabled(cache_enabled);
                
                // Perform some operations under autocast
                torch::Tensor result = torch::matmul(safe_tensor, safe_tensor);
                (void)result.sum().item<double>();
                
                // Test autocast state
                bool is_enabled = at::autocast::is_autocast_enabled(device_type);
                torch::ScalarType current_dtype = at::autocast::get_autocast_dtype(device_type);
                
                // Test autocast_increment_nesting
                at::autocast::increment_nesting();
                at::autocast::decrement_nesting();
                
                // Test set_autocast_enabled
                at::autocast::set_autocast_enabled(device_type, false);
                at::autocast::set_autocast_enabled(device_type, true);
                
                // Test set_autocast_dtype
                at::autocast::set_autocast_dtype(device_type, torch::kFloat);
                at::autocast::set_autocast_dtype(device_type, dtype);
                
                // Test clear_autocast_cache
                at::autocast::clear_cache();

                // Restore prior autocast settings
                at::autocast::set_autocast_cache_enabled(prev_cache);
                at::autocast::set_autocast_dtype(device_type, prev_dtype);
                at::autocast::set_autocast_enabled(device_type, prev_enabled);
                (void)is_enabled;
                (void)current_dtype;
            }
            
            // Test basic tensor operations that might benefit from AMP
            {
                // Create some tensors for testing
                torch::Tensor a = torch::randn({10, 10});
                torch::Tensor b = torch::randn({10, 10});
                
                // Test matrix multiplication
                torch::Tensor c = torch::matmul(a, b);
                
                // Test convolution if possible
                torch::Tensor conv_input = torch::randn({1, 3, 32, 32});
                torch::Tensor conv_weight = torch::randn({16, 3, 3, 3});
                torch::Tensor conv_output = torch::conv2d(conv_input, conv_weight);
                
                // Test linear layer
                torch::nn::Linear linear(10, 5);
                torch::Tensor linear_input = torch::randn({1, 10});
                torch::Tensor linear_output = linear(linear_input);
                
                // Test loss computation
                torch::Tensor target = torch::randint(0, 5, {1});
                torch::Tensor loss = torch::cross_entropy_loss(linear_output, target);
                
                // Test backward pass
                loss.backward();
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
