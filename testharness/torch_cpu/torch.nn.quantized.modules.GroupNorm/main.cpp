#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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

        // Extract number of groups (1-32)
        uint8_t num_groups_raw = Data[offset++];
        int64_t num_groups = (num_groups_raw % 32) + 1;

        // Extract channel multiplier (1-8) - num_channels = num_groups * multiplier
        uint8_t channel_mult = Data[offset++];
        int64_t num_channels = num_groups * ((channel_mult % 8) + 1);

        // Extract batch size (1-4)
        uint8_t batch_raw = Data[offset++];
        int64_t batch_size = (batch_raw % 4) + 1;

        // Extract spatial dimensions (1-16)
        uint8_t spatial_raw = Data[offset++];
        int64_t spatial_size = (spatial_raw % 16) + 1;

        // Extract affine flag
        bool affine = (Data[offset++] & 0x01) != 0;

        // Extract epsilon modifier
        uint8_t eps_raw = Data[offset++];
        double eps = 1e-5 * (1.0 + (eps_raw % 100));

        // Create input tensor with shape [N, C, *] where C is divisible by num_groups
        torch::Tensor input_float;
        if (offset + 4 <= Size) {
            // Use remaining data to create tensor values
            size_t remaining = Size - offset;
            int64_t num_elements = batch_size * num_channels * spatial_size;
            
            std::vector<float> values(num_elements);
            for (int64_t i = 0; i < num_elements; i++) {
                size_t idx = offset + (i % remaining);
                // Scale to reasonable range [-10, 10]
                values[i] = ((Data[idx] / 255.0f) - 0.5f) * 20.0f;
            }
            input_float = torch::from_blob(values.data(), {batch_size, num_channels, spatial_size}, 
                                           torch::kFloat).clone();
        } else {
            input_float = torch::randn({batch_size, num_channels, spatial_size});
        }

        // Create weight and bias tensors if affine
        c10::optional<torch::Tensor> weight = c10::nullopt;
        c10::optional<torch::Tensor> bias = c10::nullopt;
        
        if (affine) {
            weight = torch::ones({num_channels});
            bias = torch::zeros({num_channels});
            
            // Optionally perturb weight and bias based on input data
            if (offset < Size) {
                float weight_scale = 0.5f + (Data[offset % Size] / 255.0f);
                weight = weight.value() * weight_scale;
            }
        }

        // Test 1: Standard group_norm on float tensor
        torch::Tensor output_float;
        try {
            output_float = torch::group_norm(input_float, num_groups, weight, bias, eps);
            
            if (output_float.numel() > 0) {
                volatile float sum = output_float.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors - continue
        }

        // Test 2: Quantize input, dequantize, then apply group_norm
        // This simulates the quantized module behavior
        try {
            float scale = 0.1f;
            int zero_point = 128;
            
            torch::Tensor quantized_input = torch::quantize_per_tensor(
                input_float, scale, zero_point, torch::kQUInt8);
            
            torch::Tensor dequantized = quantized_input.dequantize();
            
            torch::Tensor output_from_quant = torch::group_norm(
                dequantized, num_groups, weight, bias, eps);
            
            if (output_from_quant.numel() > 0) {
                volatile float sum = output_from_quant.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected quantization errors - continue
        }

        // Test 3: Different input dimensions (2D, 4D)
        try {
            // 2D input: [N, C]
            torch::Tensor input_2d = input_float.select(2, 0);
            torch::Tensor output_2d = torch::group_norm(input_2d, num_groups, weight, bias, eps);
            
            if (output_2d.numel() > 0) {
                volatile float sum = output_2d.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected for certain shapes
        }

        // Test 4: 4D input [N, C, H, W]
        try {
            int64_t hw = std::max((int64_t)1, (int64_t)std::sqrt(spatial_size));
            torch::Tensor input_4d = torch::randn({batch_size, num_channels, hw, hw});
            
            torch::Tensor output_4d = torch::group_norm(input_4d, num_groups, weight, bias, eps);
            
            if (output_4d.numel() > 0) {
                volatile float sum = output_4d.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected for certain configurations
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}