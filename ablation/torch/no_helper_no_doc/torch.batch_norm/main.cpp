#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Minimum size check
        if (Size < 32) return 0;

        // Extract tensor dimensions
        auto batch_size = extract_int(Data, Size, offset, 1, 8);
        auto channels = extract_int(Data, Size, offset, 1, 16);
        auto height = extract_int(Data, Size, offset, 1, 32);
        auto width = extract_int(Data, Size, offset, 1, 32);

        // Extract batch norm parameters
        auto eps = extract_float(Data, Size, offset, 1e-8f, 1e-3f);
        auto momentum = extract_float(Data, Size, offset, 0.01f, 0.99f);
        auto training = extract_bool(Data, Size, offset);
        auto track_running_stats = extract_bool(Data, Size, offset);

        // Extract dtype and device info
        auto dtype_idx = extract_int(Data, Size, offset, 0, 2);
        auto device_idx = extract_int(Data, Size, offset, 0, 1);

        // Map to actual types
        torch::ScalarType dtype = torch::kFloat32;
        if (dtype_idx == 1) dtype = torch::kFloat64;
        else if (dtype_idx == 2) dtype = torch::kFloat16;

        torch::Device device = torch::kCPU;
        if (device_idx == 1 && torch::cuda::is_available()) {
            device = torch::kCUDA;
        }

        // Create input tensor with various shapes
        torch::Tensor input;
        auto shape_type = extract_int(Data, Size, offset, 0, 3);
        
        if (shape_type == 0) {
            // 2D: (N, C)
            input = torch::randn({batch_size, channels}, torch::TensorOptions().dtype(dtype).device(device));
        } else if (shape_type == 1) {
            // 3D: (N, C, L)
            input = torch::randn({batch_size, channels, height}, torch::TensorOptions().dtype(dtype).device(device));
        } else if (shape_type == 2) {
            // 4D: (N, C, H, W)
            input = torch::randn({batch_size, channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));
        } else {
            // 5D: (N, C, D, H, W)
            auto depth = extract_int(Data, Size, offset, 1, 16);
            input = torch::randn({batch_size, channels, depth, height, width}, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Create weight and bias tensors
        torch::Tensor weight, bias;
        auto use_weight = extract_bool(Data, Size, offset);
        auto use_bias = extract_bool(Data, Size, offset);

        if (use_weight) {
            weight = torch::randn({channels}, torch::TensorOptions().dtype(dtype).device(device));
        }
        if (use_bias) {
            bias = torch::randn({channels}, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Create running mean and var tensors
        torch::Tensor running_mean, running_var;
        if (track_running_stats) {
            running_mean = torch::zeros({channels}, torch::TensorOptions().dtype(dtype).device(device));
            running_var = torch::ones({channels}, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Test different batch norm variants
        auto test_variant = extract_int(Data, Size, offset, 0, 4);

        if (test_variant == 0) {
            // Basic batch_norm call
            auto result = torch::batch_norm(input, weight, bias, running_mean, running_var, 
                                          training, momentum, eps, torch::backends::cudnn::enabled());
        } else if (test_variant == 1) {
            // Functional batch_norm
            auto result = torch::nn::functional::batch_norm(input, running_mean, running_var,
                torch::nn::functional::BatchNormFuncOptions()
                    .weight(weight)
                    .bias(bias)
                    .training(training)
                    .momentum(momentum)
                    .eps(eps));
        } else if (test_variant == 2) {
            // Test with extreme values
            auto extreme_input = input.clone();
            if (extract_bool(Data, Size, offset)) {
                extreme_input = extreme_input * 1e6f; // Large values
            } else {
                extreme_input = extreme_input * 1e-6f; // Small values
            }
            auto result = torch::batch_norm(extreme_input, weight, bias, running_mean, running_var,
                                          training, momentum, eps, torch::backends::cudnn::enabled());
        } else if (test_variant == 3) {
            // Test with NaN/Inf handling
            auto nan_input = input.clone();
            if (extract_bool(Data, Size, offset)) {
                nan_input[0][0] = std::numeric_limits<float>::quiet_NaN();
            }
            if (extract_bool(Data, Size, offset)) {
                nan_input[0][1 % channels] = std::numeric_limits<float>::infinity();
            }
            auto result = torch::batch_norm(nan_input, weight, bias, running_mean, running_var,
                                          training, momentum, eps, torch::backends::cudnn::enabled());
        } else {
            // Test with different momentum and eps edge cases
            auto edge_momentum = extract_bool(Data, Size, offset) ? 0.0 : 1.0;
            auto edge_eps = extract_bool(Data, Size, offset) ? 0.0 : 1.0;
            auto result = torch::batch_norm(input, weight, bias, running_mean, running_var,
                                          training, edge_momentum, edge_eps, torch::backends::cudnn::enabled());
        }

        // Test gradient computation if training
        if (training && input.requires_grad()) {
            input.requires_grad_(true);
            if (weight.defined()) weight.requires_grad_(true);
            if (bias.defined()) bias.requires_grad_(true);
            
            auto output = torch::batch_norm(input, weight, bias, running_mean, running_var,
                                          training, momentum, eps, torch::backends::cudnn::enabled());
            auto loss = output.sum();
            loss.backward();
        }

        // Test with different tensor layouts
        if (extract_bool(Data, Size, offset) && input.dim() == 4) {
            auto channels_last_input = input.to(torch::MemoryFormat::ChannelsLast);
            auto result = torch::batch_norm(channels_last_input, weight, bias, running_mean, running_var,
                                          training, momentum, eps, torch::backends::cudnn::enabled());
        }

        // Test batch norm with zero-sized dimensions
        if (extract_bool(Data, Size, offset)) {
            auto zero_batch = torch::randn({0, channels, height, width}, torch::TensorOptions().dtype(dtype).device(device));
            auto result = torch::batch_norm(zero_batch, weight, bias, running_mean, running_var,
                                          training, momentum, eps, torch::backends::cudnn::enabled());
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}