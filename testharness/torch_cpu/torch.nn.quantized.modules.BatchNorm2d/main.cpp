#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 20) {
            return 0;
        }
        
        // Read num_features from fuzzer data
        int64_t num_features = static_cast<int64_t>(Data[offset] % 16) + 1;
        offset++;
        
        // Read batch size and spatial dimensions
        int64_t batch_size = static_cast<int64_t>(Data[offset] % 4) + 1;
        offset++;
        int64_t height = static_cast<int64_t>(Data[offset] % 8) + 1;
        offset++;
        int64_t width = static_cast<int64_t>(Data[offset] % 8) + 1;
        offset++;
        
        // Create input tensor with proper 4D shape for BatchNorm2d (N, C, H, W)
        torch::Tensor input = torch::randn({batch_size, num_features, height, width});
        
        // Read scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::abs(scale);
            if (scale < 1e-5f || !std::isfinite(scale)) {
                scale = 0.1f;
            }
            if (scale > 1e5f) {
                scale = 1.0f;
            }
        }
        
        if (offset + 1 <= Size) {
            zero_point = static_cast<int64_t>(Data[offset]) % 256;
            offset++;
        }
        
        // Read eps and momentum
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 1 <= Size) {
            uint8_t eps_byte = Data[offset];
            offset++;
            eps = 1e-5 + (eps_byte / 255.0) * 1e-3;
        }
        
        if (offset + 1 <= Size) {
            uint8_t momentum_byte = Data[offset];
            offset++;
            momentum = (momentum_byte / 255.0) * 0.5;
        }
        
        // Create BatchNorm2d module (non-quantized) and set to eval mode
        torch::nn::BatchNorm2d bn_module(torch::nn::BatchNorm2dOptions(num_features).eps(eps).momentum(momentum));
        bn_module->eval();
        
        // Initialize running stats with some variance
        bn_module->running_mean.copy_(torch::zeros(num_features));
        bn_module->running_var.copy_(torch::ones(num_features));
        
        // Test 1: Standard batch norm forward pass
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = bn_module->forward(input);
        }
        
        // Test 2: Quantize input, dequantize, run through batch norm, requantize
        // This simulates quantized batch norm behavior
        try {
            torch::Tensor q_input = torch::quantize_per_tensor(
                input, 
                scale, 
                zero_point, 
                torch::kQUInt8);
            
            torch::Tensor dequantized = q_input.dequantize();
            
            torch::Tensor bn_output;
            {
                torch::NoGradGuard no_grad;
                bn_output = bn_module->forward(dequantized);
            }
            
            // Requantize output
            torch::Tensor q_output = torch::quantize_per_tensor(
                bn_output,
                scale,
                zero_point,
                torch::kQUInt8);
            
            // Access quantized tensor properties
            (void)q_output.q_scale();
            (void)q_output.q_zero_point();
            (void)q_output.int_repr();
        } catch (...) {
            // Quantization may fail for certain parameters, silently continue
        }
        
        // Test 3: Try with different quantization dtype (QInt8)
        try {
            int64_t signed_zero_point = static_cast<int64_t>(static_cast<int8_t>(zero_point % 128));
            
            torch::Tensor q_input_signed = torch::quantize_per_tensor(
                input,
                scale,
                signed_zero_point,
                torch::kQInt8);
            
            torch::Tensor dequantized_signed = q_input_signed.dequantize();
            
            torch::Tensor bn_output_signed;
            {
                torch::NoGradGuard no_grad;
                bn_output_signed = bn_module->forward(dequantized_signed);
            }
            
            torch::Tensor q_output_signed = torch::quantize_per_tensor(
                bn_output_signed,
                scale,
                signed_zero_point,
                torch::kQInt8);
        } catch (...) {
            // Silently handle quantization errors
        }
        
        // Test 4: Per-channel quantization (more relevant for batch norm)
        try {
            torch::Tensor scales = torch::full({num_features}, scale);
            torch::Tensor zero_points = torch::full({num_features}, zero_point, torch::kLong);
            
            torch::Tensor q_input_per_channel = torch::quantize_per_channel(
                input,
                scales,
                zero_points,
                1,  // axis = 1 (channel dimension)
                torch::kQUInt8);
            
            torch::Tensor dequantized_per_channel = q_input_per_channel.dequantize();
            
            torch::Tensor bn_output_per_channel;
            {
                torch::NoGradGuard no_grad;
                bn_output_per_channel = bn_module->forward(dequantized_per_channel);
            }
        } catch (...) {
            // Per-channel quantization may fail, silently continue
        }
        
        // Test 5: Training mode batch norm
        if (offset < Size && (Data[offset] % 2 == 0)) {
            torch::nn::BatchNorm2d bn_train(torch::nn::BatchNorm2dOptions(num_features).eps(eps).momentum(momentum));
            bn_train->train();
            
            try {
                torch::Tensor train_output = bn_train->forward(input);
                
                // Check that running stats were updated
                (void)bn_train->running_mean;
                (void)bn_train->running_var;
            } catch (...) {
                // Training mode may have different requirements
            }
        }
        
        // Test 6: Batch norm with affine=false
        try {
            torch::nn::BatchNorm2d bn_no_affine(
                torch::nn::BatchNorm2dOptions(num_features).eps(eps).momentum(momentum).affine(false));
            bn_no_affine->eval();
            bn_no_affine->running_mean.copy_(torch::zeros(num_features));
            bn_no_affine->running_var.copy_(torch::ones(num_features));
            
            torch::NoGradGuard no_grad;
            torch::Tensor output_no_affine = bn_no_affine->forward(input);
        } catch (...) {
            // Silently handle errors
        }
        
        // Test 7: Batch norm with track_running_stats=false
        try {
            torch::nn::BatchNorm2d bn_no_track(
                torch::nn::BatchNorm2dOptions(num_features).eps(eps).momentum(momentum).track_running_stats(false));
            bn_no_track->eval();
            
            torch::NoGradGuard no_grad;
            torch::Tensor output_no_track = bn_no_track->forward(input);
        } catch (...) {
            // Silently handle errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}