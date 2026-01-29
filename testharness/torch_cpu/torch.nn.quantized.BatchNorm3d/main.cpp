#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/ATen.h>

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
        
        // Early exit if not enough data
        if (Size < 20) {
            return 0;
        }
        
        // Create input tensor - must be 5D for BatchNorm3d: (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor is 5D (required for BatchNorm3d)
        if (input.dim() != 5) {
            std::vector<int64_t> new_shape;
            if (input.dim() < 5) {
                new_shape = input.sizes().vec();
                while (new_shape.size() < 5) {
                    new_shape.push_back(1);
                }
            } else {
                // Collapse extra dimensions into spatial dimensions
                int64_t total = input.numel();
                new_shape = {1, 1, 1, 1, total};
            }
            try {
                input = input.reshape(new_shape);
            } catch (...) {
                return 0;
            }
        }
        
        // Ensure input is float and contiguous for quantization
        input = input.to(torch::kFloat32).contiguous();
        
        // Get the number of channels (second dimension)
        int64_t num_features = input.size(1);
        if (num_features <= 0) {
            return 0;
        }
        
        // Parse additional parameters from the input data
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 16 <= Size) {
            uint32_t raw_eps_int;
            std::memcpy(&raw_eps_int, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            eps = 1e-10 + (raw_eps_int % 1000) * 1e-5;
            
            uint32_t raw_momentum_int;
            std::memcpy(&raw_momentum_int, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            momentum = (raw_momentum_int % 100) / 100.0;
        }
        
        // Create running_mean and running_var
        auto running_mean = torch::zeros({num_features}, torch::kFloat32);
        auto running_var = torch::ones({num_features}, torch::kFloat32);
        
        // Create weight and bias
        auto weight = torch::ones({num_features}, torch::kFloat32);
        auto bias = torch::zeros({num_features}, torch::kFloat32);
        
        // Apply regular batch_norm first (since quantized_batch_norm3d module is not directly available in C++ API)
        // We test the underlying batch_norm operation that would be used
        auto output = torch::batch_norm(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            false,  // training
            momentum,
            eps,
            false   // cudnn_enabled
        );
        
        // Now test quantization path
        // Quantize input per-tensor (simpler and more reliable)
        double scale = 0.1;
        int64_t zero_point = 128;
        
        if (offset + 8 <= Size) {
            uint32_t scale_raw;
            std::memcpy(&scale_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            scale = 0.001 + (scale_raw % 1000) * 0.001;
            
            uint8_t zp_raw;
            std::memcpy(&zp_raw, Data + offset, sizeof(uint8_t));
            offset += sizeof(uint8_t);
            zero_point = zp_raw;
        }
        
        try {
            // Quantize the input tensor per-tensor
            auto q_input = torch::quantize_per_tensor(
                input,
                scale,
                zero_point,
                torch::kQUInt8
            );
            
            // Dequantize to verify the quantization worked
            auto dq_input = q_input.dequantize();
            
            // Apply batch norm on dequantized tensor (simulating quantized batch norm behavior)
            auto q_output = torch::batch_norm(
                dq_input,
                weight,
                bias,
                running_mean,
                running_var,
                false,
                momentum,
                eps,
                false
            );
            
            // Re-quantize output
            auto final_q = torch::quantize_per_tensor(q_output, scale, zero_point, torch::kQUInt8);
            auto final_output = final_q.dequantize();
            
        } catch (...) {
            // Quantization may fail for certain tensor configurations, that's expected
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}