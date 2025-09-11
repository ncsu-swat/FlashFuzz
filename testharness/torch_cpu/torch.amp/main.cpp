#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Test torch.amp.autocast
        if (offset < Size) {
            bool enabled = Data[offset++] % 2 == 0;
            
            // Get device type
            torch::DeviceType device_type = torch::kCPU;
            if (offset < Size) {
                uint8_t device_selector = Data[offset++];
                device_type = (device_selector % 2 == 0) ? torch::kCPU : torch::kCUDA;
            }
            
            // Get dtype
            torch::ScalarType dtype = torch::kFloat;
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
            
            // Test autocast using at::autocast
            {
                at::autocast::AutocastMode autocast_mode(device_type, enabled, dtype, cache_enabled);
                
                // Perform some operations under autocast
                torch::Tensor result = torch::matmul(input_tensor, input_tensor);
                
                // Test autocast state
                bool is_enabled = at::autocast::is_enabled();
                torch::DeviceType current_device = at::autocast::get_autocast_device_type();
                torch::ScalarType current_dtype = at::autocast::get_autocast_dtype(device_type);
                
                // Test autocast_increment_nesting
                at::autocast::increment_nesting();
                at::autocast::decrement_nesting();
                
                // Test set_autocast_enabled
                at::autocast::set_enabled(device_type, false);
                at::autocast::set_enabled(device_type, true);
                
                // Test set_autocast_dtype
                at::autocast::set_autocast_dtype(device_type, torch::kFloat);
                at::autocast::set_autocast_dtype(device_type, dtype);
                
                // Test clear_autocast_cache
                at::autocast::clear_cache();
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
