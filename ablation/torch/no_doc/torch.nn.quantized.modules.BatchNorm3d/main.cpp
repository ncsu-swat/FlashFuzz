#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 20) {
            // Need minimum bytes for basic parsing
            return 0;
        }

        size_t offset = 0;

        // Parse configuration parameters from fuzzer input
        uint8_t config_byte1 = (offset < Size) ? Data[offset++] : 0;
        uint8_t config_byte2 = (offset < Size) ? Data[offset++] : 0;
        uint8_t config_byte3 = (offset < Size) ? Data[offset++] : 0;
        uint8_t config_byte4 = (offset < Size) ? Data[offset++] : 0;
        
        // Determine number of features (channels) - important for BatchNorm3d
        int64_t num_features = 1 + (config_byte1 % 256);
        
        // Parse eps value (small positive value for numerical stability)
        double eps = 1e-8 + (config_byte2 % 100) * 1e-7;
        
        // Parse momentum value (between 0 and 1)
        double momentum = (config_byte3 % 100) / 100.0;
        
        // Parse affine flag (whether to learn gamma and beta)
        bool affine = config_byte4 & 0x01;
        
        // Parse track_running_stats flag
        bool track_running_stats = config_byte4 & 0x02;
        
        // Create quantized BatchNorm3d module
        torch::nn::BatchNorm3dOptions options(num_features);
        options.eps(eps);
        options.momentum(momentum);
        options.affine(affine);
        options.track_running_stats(track_running_stats);
        
        auto bn3d = torch::nn::BatchNorm3d(options);
        
        // Parse scale and zero_point for quantization
        float scale = 0.01f + (offset < Size ? Data[offset++] : 1) / 255.0f;
        int32_t zero_point = offset < Size ? static_cast<int32_t>(Data[offset++]) - 128 : 0;
        
        // Create input tensor from fuzzer data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (...) {
            // If tensor creation fails, create a default one
            input = torch::randn({2, num_features, 4, 4, 4});
        }
        
        // Ensure input is 5D for BatchNorm3d (N, C, D, H, W)
        if (input.dim() != 5) {
            // Reshape or create new tensor with proper dimensions
            int64_t total_elements = input.numel();
            if (total_elements < num_features * 8) {
                // Not enough elements, create random tensor
                input = torch::randn({1, num_features, 2, 2, 2});
            } else {
                // Try to reshape to 5D
                int64_t batch_size = std::max(int64_t(1), total_elements / (num_features * 64));
                int64_t depth = 4;
                int64_t height = 4;
                int64_t width = std::max(int64_t(1), total_elements / (batch_size * num_features * depth * height));
                
                if (batch_size * num_features * depth * height * width != total_elements) {
                    // Adjust dimensions to match total elements
                    batch_size = 1;
                    depth = 2;
                    height = 2;
                    width = std::max(int64_t(1), total_elements / (num_features * depth * height));
                    if (num_features * depth * height * width > total_elements) {
                        input = torch::randn({1, num_features, 2, 2, 2});
                    } else {
                        input = input.view({batch_size, num_features, depth, height, width});
                    }
                } else {
                    input = input.view({batch_size, num_features, depth, height, width});
                }
            }
        }
        
        // Ensure the channel dimension matches num_features
        if (input.size(1) != num_features) {
            int64_t batch = input.size(0);
            input = torch::randn({batch, num_features, 4, 4, 4});
        }
        
        // Convert to float for processing (BatchNorm typically works with float)
        if (input.dtype() != torch::kFloat32) {
            input = input.to(torch::kFloat32);
        }
        
        // Set module to training or eval mode based on fuzzer input
        bool training_mode = (offset < Size) ? (Data[offset++] & 0x01) : false;
        if (training_mode) {
            bn3d->train();
        } else {
            bn3d->eval();
        }
        
        // Initialize running stats if needed
        if (track_running_stats && !bn3d->running_mean.defined()) {
            bn3d->running_mean = torch::zeros({num_features});
            bn3d->running_var = torch::ones({num_features});
        }
        
        // Apply batch normalization
        torch::Tensor output = bn3d->forward(input);
        
        // Quantize the output
        torch::Tensor quantized_output = torch::quantize_per_tensor(
            output, scale, zero_point, torch::kQUInt8
        );
        
        // Dequantize for verification
        torch::Tensor dequantized = quantized_output.dequantize();
        
        // Additional operations to increase coverage
        
        // Test with different quantization schemes
        if (offset < Size && Data[offset++] & 0x01) {
            // Per-channel quantization
            auto scales = torch::rand({num_features}) * 0.5 + 0.01;
            auto zero_points = torch::randint(-128, 127, {num_features}, torch::kInt32);
            
            try {
                auto per_channel_quantized = torch::quantize_per_channel(
                    output, scales, zero_points, 1, torch::kQInt8
                );
                auto per_channel_dequantized = per_channel_quantized.dequantize();
                
                // Compare shapes
                if (per_channel_dequantized.sizes() != output.sizes()) {
                    std::cerr << "Shape mismatch after per-channel quantization" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Some configurations might not be valid for per-channel quantization
            }
        }
        
        // Test batch norm with different input shapes (edge cases)
        if (offset < Size && Data[offset++] & 0x01) {
            // Test with batch size 1
            auto single_batch = torch::randn({1, num_features, 3, 3, 3});
            auto single_output = bn3d->forward(single_batch);
            
            // Test with minimal spatial dimensions
            auto minimal_spatial = torch::randn({2, num_features, 1, 1, 1});
            auto minimal_output = bn3d->forward(minimal_spatial);
        }
        
        // Test gradient computation if in training mode
        if (training_mode && input.requires_grad()) {
            try {
                auto loss = output.mean();
                loss.backward();
                
                // Check if gradients are computed
                if (bn3d->weight.defined() && affine) {
                    auto weight_grad = bn3d->weight.grad();
                    if (weight_grad.defined()) {
                        // Gradient exists, operation successful
                    }
                }
            } catch (const c10::Error& e) {
                // Gradient computation might fail for some configurations
            }
        }
        
        // Test state dict operations
        try {
            auto state_dict = bn3d->named_parameters();
            for (const auto& pair : state_dict) {
                auto param_name = pair.key();
                auto param_tensor = pair.value();
                
                // Quantize parameters if they exist
                if (param_tensor.defined() && param_tensor.numel() > 0) {
                    auto quantized_param = torch::quantize_per_tensor(
                        param_tensor, 0.1, 0, torch::kQInt8
                    );
                }
            }
        } catch (...) {
            // State dict operations might fail
        }
        
        // Test with zero-dimensional edge cases
        if (offset < Size && Data[offset++] & 0x01) {
            try {
                // Create tensor with one dimension being 0
                auto zero_dim_input = torch::randn({2, num_features, 0, 4, 4});
                if (zero_dim_input.numel() == 0) {
                    // Skip forward pass for zero-element tensors
                } else {
                    auto zero_dim_output = bn3d->forward(zero_dim_input);
                }
            } catch (...) {
                // Zero-dimensional inputs might cause issues
            }
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors
        return 0; // Continue fuzzing
    }
    catch (const std::bad_alloc &e)
    {
        // Memory allocation failures
        return 0; // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Discard this input
    }
    catch (...)
    {
        // Unknown exceptions
        return -1; // Discard this input
    }
}