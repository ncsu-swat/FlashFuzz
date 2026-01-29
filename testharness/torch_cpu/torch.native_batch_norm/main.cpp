#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor and ensure it's float type
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Batch norm requires at least 2D input (N, C, ...)
        if (input.dim() < 2) {
            // Reshape to at least 2D
            int64_t total_elements = input.numel();
            if (total_elements < 1) {
                return 0;
            }
            input = input.reshape({1, total_elements});
        }
        
        // Ensure float type for batch norm
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // For batch norm, num_features is the channel dimension (dim=1)
        int64_t num_features = input.size(1);
        
        if (num_features < 1) {
            return 0;
        }
        
        // Create weight and bias with appropriate size
        torch::Tensor weight = torch::ones({num_features}, torch::kFloat32);
        torch::Tensor bias = torch::zeros({num_features}, torch::kFloat32);
        
        // Create running mean and running var tensors
        torch::Tensor running_mean = torch::zeros({num_features}, torch::kFloat32);
        torch::Tensor running_var = torch::ones({num_features}, torch::kFloat32);
        
        // Get training mode from input data if available
        bool training = true;
        if (offset < Size) {
            training = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Get momentum from input data if available
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float momentum_f;
            std::memcpy(&momentum_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure momentum is in valid range [0, 1]
            momentum = std::abs(static_cast<double>(momentum_f));
            if (std::isnan(momentum) || std::isinf(momentum)) {
                momentum = 0.1;
            } else {
                momentum = momentum - std::floor(momentum); // Clamp to [0, 1)
            }
        }
        
        // Get epsilon from input data if available
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure epsilon is positive and reasonable
            eps = std::abs(static_cast<double>(eps_f));
            if (std::isnan(eps) || std::isinf(eps) || eps < 1e-10) {
                eps = 1e-5;
            }
            if (eps > 1.0) {
                eps = 1e-5;
            }
        }
        
        // Test 1: Basic native_batch_norm call
        try {
            auto result = torch::native_batch_norm(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps
            );
            
            torch::Tensor output = std::get<0>(result);
            torch::Tensor save_mean = std::get<1>(result);
            torch::Tensor save_var = std::get<2>(result);
            
            // Access to ensure computation happens
            if (output.defined()) {
                volatile float val = output.sum().item<float>();
                (void)val;
            }
        } catch (const c10::Error&) {
            // Expected failures (shape issues, etc.) - ignore silently
        }
        
        // Test 2: With optional weight/bias as None (using empty tensors)
        try {
            auto result2 = torch::native_batch_norm(
                input,
                torch::Tensor(),  // No weight
                torch::Tensor(),  // No bias
                running_mean,
                running_var,
                training,
                momentum,
                eps
            );
            
            torch::Tensor output2 = std::get<0>(result2);
            if (output2.defined()) {
                volatile float val = output2.sum().item<float>();
                (void)val;
            }
        } catch (const c10::Error&) {
            // Expected failures - ignore silently
        }
        
        // Test 3: Inference mode (training=false)
        try {
            auto result3 = torch::native_batch_norm(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                false,  // inference mode
                momentum,
                eps
            );
            
            torch::Tensor output3 = std::get<0>(result3);
            if (output3.defined()) {
                volatile float val = output3.sum().item<float>();
                (void)val;
            }
        } catch (const c10::Error&) {
            // Expected failures - ignore silently
        }
        
        // Test 4: Training mode without running stats
        try {
            auto result4 = torch::native_batch_norm(
                input,
                weight,
                bias,
                torch::Tensor(),  // No running mean
                torch::Tensor(),  // No running var
                true,  // training mode
                momentum,
                eps
            );
            
            torch::Tensor output4 = std::get<0>(result4);
            if (output4.defined()) {
                volatile float val = output4.sum().item<float>();
                (void)val;
            }
        } catch (const c10::Error&) {
            // Expected failures - ignore silently
        }
        
        // Test 5: Different input shapes if we have more data
        if (offset + 4 <= Size) {
            try {
                int8_t batch_size = static_cast<int8_t>(Data[offset++] % 8) + 1;
                int8_t channels = static_cast<int8_t>(Data[offset++] % 8) + 1;
                int8_t height = static_cast<int8_t>(Data[offset++] % 8) + 1;
                int8_t width = static_cast<int8_t>(Data[offset++] % 8) + 1;
                
                torch::Tensor input_4d = torch::randn({batch_size, channels, height, width}, torch::kFloat32);
                torch::Tensor weight_4d = torch::ones({channels}, torch::kFloat32);
                torch::Tensor bias_4d = torch::zeros({channels}, torch::kFloat32);
                torch::Tensor running_mean_4d = torch::zeros({channels}, torch::kFloat32);
                torch::Tensor running_var_4d = torch::ones({channels}, torch::kFloat32);
                
                auto result5 = torch::native_batch_norm(
                    input_4d,
                    weight_4d,
                    bias_4d,
                    running_mean_4d,
                    running_var_4d,
                    training,
                    momentum,
                    eps
                );
                
                torch::Tensor output5 = std::get<0>(result5);
                if (output5.defined()) {
                    volatile float val = output5.sum().item<float>();
                    (void)val;
                }
            } catch (const c10::Error&) {
                // Expected failures - ignore silently
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}