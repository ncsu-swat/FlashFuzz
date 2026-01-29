#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters for BatchNorm
        uint8_t num_features_byte = Data[offset++];
        int64_t num_features = (num_features_byte % 32) + 1;  // 1-32 channels
        
        // Extract batch size
        uint8_t batch_byte = Data[offset++];
        int64_t batch_size = (batch_byte % 4) + 1;  // 1-4
        
        // Extract spatial dimensions
        uint8_t h_byte = Data[offset++];
        uint8_t w_byte = Data[offset++];
        int64_t height = (h_byte % 8) + 1;  // 1-8
        int64_t width = (w_byte % 8) + 1;   // 1-8
        
        // Extract eps parameter
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps_f = std::abs(eps_f);
            if (std::isfinite(eps_f) && eps_f > 1e-10 && eps_f < 1.0) {
                eps = static_cast<double>(eps_f);
            }
        }
        
        // Extract momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float mom_f;
            std::memcpy(&mom_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            mom_f = std::abs(mom_f);
            if (std::isfinite(mom_f) && mom_f > 0.0f && mom_f <= 1.0f) {
                momentum = static_cast<double>(mom_f);
            }
        }
        
        // Extract affine flag
        bool affine = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Extract track_running_stats flag
        bool track_running_stats = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Create input tensor with proper shape [N, C, H, W]
        torch::Tensor input = torch::randn({batch_size, num_features, height, width}, 
                                           torch::TensorOptions().dtype(torch::kFloat32));
        
        // Use remaining fuzzer data to perturb the tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t elements = std::min(remaining / sizeof(float), 
                                       static_cast<size_t>(input.numel()));
            if (elements > 0) {
                auto accessor = input.accessor<float, 4>();
                size_t idx = 0;
                for (int64_t n = 0; n < batch_size && idx < elements; n++) {
                    for (int64_t c = 0; c < num_features && idx < elements; c++) {
                        for (int64_t h = 0; h < height && idx < elements; h++) {
                            for (int64_t w = 0; w < width && idx < elements; w++) {
                                float val;
                                std::memcpy(&val, Data + offset + idx * sizeof(float), sizeof(float));
                                if (std::isfinite(val)) {
                                    accessor[n][c][h][w] = val;
                                }
                                idx++;
                            }
                        }
                    }
                }
            }
        }
        
        // Create BatchNorm2d module 
        // Note: torch::nn::quantized::BatchNorm is not available in C++ frontend,
        // so we simulate quantized batchnorm using regular batchnorm with quantization
        torch::nn::BatchNorm2d batchnorm(
            torch::nn::BatchNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Set to eval mode for consistent behavior
        batchnorm->eval();
        
        // Test 1: Regular BatchNorm forward pass
        try {
            torch::Tensor output = batchnorm(input);
            (void)output;
        } catch (const c10::Error&) {
            // Shape/type mismatch - expected
        }
        
        // Test 2: Quantized workflow - quantize input, process, quantize output
        try {
            float scale = 1.0f / 128.0f;
            int64_t zero_point = 128;
            
            // Quantize input tensor
            torch::Tensor quantized_input = torch::quantize_per_tensor(
                input, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
            
            // Dequantize for BatchNorm processing
            torch::Tensor dequantized_input = torch::dequantize(quantized_input);
            
            // Apply BatchNorm
            torch::Tensor output = batchnorm(dequantized_input);
            
            // Quantize output
            torch::Tensor quantized_output = torch::quantize_per_tensor(
                output, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
            
            (void)quantized_output;
        } catch (const c10::Error&) {
            // Quantization errors - expected for some inputs
        }
        
        // Test 3: Per-channel quantization
        try {
            torch::Tensor scales = torch::ones({num_features}) / 128.0f;
            torch::Tensor zero_points = torch::full({num_features}, 128, torch::kLong);
            
            torch::Tensor quantized_per_channel = torch::quantize_per_channel(
                input,
                scales,
                zero_points,
                1,  // axis for channels
                torch::kQUInt8
            );
            
            torch::Tensor dequantized = torch::dequantize(quantized_per_channel);
            torch::Tensor output = batchnorm(dequantized);
            
            (void)output;
        } catch (const c10::Error&) {
            // Per-channel quantization errors - expected
        }
        
        // Test 4: Training mode
        try {
            batchnorm->train();
            torch::Tensor train_output = batchnorm(input);
            (void)train_output;
        } catch (const c10::Error&) {
            // Training mode errors - expected
        }
        
        // Test 5: BatchNorm1d with 2D input
        try {
            torch::nn::BatchNorm1d batchnorm1d(
                torch::nn::BatchNorm1dOptions(num_features)
                    .eps(eps)
                    .momentum(momentum)
            );
            batchnorm1d->eval();
            
            torch::Tensor input_1d = torch::randn({batch_size, num_features});
            torch::Tensor output_1d = batchnorm1d(input_1d);
            (void)output_1d;
        } catch (const c10::Error&) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}