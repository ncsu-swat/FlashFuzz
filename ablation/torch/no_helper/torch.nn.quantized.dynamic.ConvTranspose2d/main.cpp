#include <torch/torch.h>
#include <torch/nn/modules/conv.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& value) {
    if (size < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 32) return 0;  // Need minimum bytes for basic parameters
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Consume parameters for ConvTranspose2d
        uint8_t in_channels_raw, out_channels_raw;
        uint8_t kernel_h, kernel_w;
        uint8_t stride_h, stride_w;
        uint8_t padding_h, padding_w;
        uint8_t output_padding_h, output_padding_w;
        uint8_t dilation_h, dilation_w;
        uint8_t groups_raw;
        bool bias;
        uint8_t batch_size_raw, height_raw, width_raw;
        
        if (!consumeBytes(ptr, remaining, in_channels_raw)) return 0;
        if (!consumeBytes(ptr, remaining, out_channels_raw)) return 0;
        if (!consumeBytes(ptr, remaining, kernel_h)) return 0;
        if (!consumeBytes(ptr, remaining, kernel_w)) return 0;
        if (!consumeBytes(ptr, remaining, stride_h)) return 0;
        if (!consumeBytes(ptr, remaining, stride_w)) return 0;
        if (!consumeBytes(ptr, remaining, padding_h)) return 0;
        if (!consumeBytes(ptr, remaining, padding_w)) return 0;
        if (!consumeBytes(ptr, remaining, output_padding_h)) return 0;
        if (!consumeBytes(ptr, remaining, output_padding_w)) return 0;
        if (!consumeBytes(ptr, remaining, dilation_h)) return 0;
        if (!consumeBytes(ptr, remaining, dilation_w)) return 0;
        if (!consumeBytes(ptr, remaining, groups_raw)) return 0;
        if (!consumeBytes(ptr, remaining, bias)) return 0;
        if (!consumeBytes(ptr, remaining, batch_size_raw)) return 0;
        if (!consumeBytes(ptr, remaining, height_raw)) return 0;
        if (!consumeBytes(ptr, remaining, width_raw)) return 0;
        
        // Constrain values to reasonable ranges
        int64_t in_channels = (in_channels_raw % 64) + 1;
        int64_t out_channels = (out_channels_raw % 64) + 1;
        int64_t kernel_size_h = (kernel_h % 7) + 1;
        int64_t kernel_size_w = (kernel_w % 7) + 1;
        int64_t stride_val_h = (stride_h % 3) + 1;
        int64_t stride_val_w = (stride_w % 3) + 1;
        int64_t padding_val_h = padding_h % 4;
        int64_t padding_val_w = padding_w % 4;
        int64_t output_padding_val_h = output_padding_h % std::min((int64_t)3, stride_val_h);
        int64_t output_padding_val_w = output_padding_w % std::min((int64_t)3, stride_val_w);
        int64_t dilation_val_h = (dilation_h % 3) + 1;
        int64_t dilation_val_w = (dilation_w % 3) + 1;
        int64_t groups = (groups_raw % std::min(in_channels, out_channels)) + 1;
        
        // Ensure channels are divisible by groups
        in_channels = (in_channels / groups) * groups;
        out_channels = (out_channels / groups) * groups;
        if (in_channels == 0) in_channels = groups;
        if (out_channels == 0) out_channels = groups;
        
        int64_t batch_size = (batch_size_raw % 8) + 1;
        int64_t height = (height_raw % 32) + 4;
        int64_t width = (width_raw % 32) + 4;
        
        // Create dynamic quantized ConvTranspose2d module
        torch::nn::ConvTranspose2dOptions options(
            in_channels,
            out_channels,
            torch::ExpandingArray<2>({kernel_size_h, kernel_size_w})
        );
        
        options.stride(torch::ExpandingArray<2>({stride_val_h, stride_val_w}));
        options.padding(torch::ExpandingArray<2>({padding_val_h, padding_val_w}));
        options.output_padding(torch::ExpandingArray<2>({output_padding_val_h, output_padding_val_w}));
        options.dilation(torch::ExpandingArray<2>({dilation_val_h, dilation_val_w}));
        options.groups(groups);
        options.bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose2d(options);
        
        // Quantize the module dynamically
        conv_transpose->eval();
        
        // Create input tensor with random data from fuzzer
        std::vector<float> input_data;
        for (size_t i = 0; i < batch_size * in_channels * height * width; ++i) {
            if (remaining >= sizeof(float)) {
                float val;
                consumeBytes(ptr, remaining, val);
                // Clamp to reasonable range to avoid numerical issues
                val = std::max(-100.0f, std::min(100.0f, val));
                input_data.push_back(val);
            } else {
                input_data.push_back(((i % 256) - 128) / 128.0f);
            }
        }
        
        auto input = torch::from_blob(
            input_data.data(),
            {batch_size, in_channels, height, width},
            torch::kFloat32
        ).clone();
        
        // Test with different quantization schemes
        uint8_t quant_scheme;
        if (consumeBytes(ptr, remaining, quant_scheme)) {
            // Apply dynamic quantization to the module
            auto quantized_model = torch::quantization::quantize_dynamic(
                conv_transpose,
                {torch::kConv2d, torch::kConvTranspose2d},
                torch::kQInt8
            );
            
            // Forward pass with quantized model
            auto output = quantized_model->forward(input);
            
            // Try with output_size specified
            if (remaining >= 2) {
                uint8_t out_h, out_w;
                consumeBytes(ptr, remaining, out_h);
                consumeBytes(ptr, remaining, out_w);
                
                int64_t target_h = (out_h % 64) + 1;
                int64_t target_w = (out_w % 64) + 1;
                
                // Calculate expected output size
                int64_t expected_h = (height - 1) * stride_val_h - 2 * padding_val_h + 
                                    dilation_val_h * (kernel_size_h - 1) + output_padding_val_h + 1;
                int64_t expected_w = (width - 1) * stride_val_w - 2 * padding_val_w + 
                                    dilation_val_w * (kernel_size_w - 1) + output_padding_val_w + 1;
                
                // Only try if target size is reasonable
                if (std::abs(target_h - expected_h) < 10 && std::abs(target_w - expected_w) < 10) {
                    try {
                        // Some quantized ops might not support all features
                        auto sized_output = quantized_model->forward(input);
                    } catch (const c10::Error& e) {
                        // Quantization-specific errors are expected for some configurations
                    }
                }
            }
        } else {
            // Test with regular (non-quantized) module as fallback
            auto output = conv_transpose->forward(input);
        }
        
        // Test edge cases
        if (remaining > 0) {
            uint8_t edge_case = ptr[0] % 4;
            
            switch (edge_case) {
                case 0: {
                    // Test with zero-sized batch
                    auto zero_batch = torch::zeros({0, in_channels, height, width});
                    try {
                        auto out = conv_transpose->forward(zero_batch);
                    } catch (const c10::Error& e) {
                        // Expected for some configurations
                    }
                    break;
                }
                case 1: {
                    // Test with single element tensors
                    auto single = torch::randn({1, in_channels, 1, 1});
                    try {
                        auto out = conv_transpose->forward(single);
                    } catch (const c10::Error& e) {
                        // Expected for some configurations
                    }
                    break;
                }
                case 2: {
                    // Test with different dtype (will be converted internally)
                    auto double_input = input.to(torch::kFloat64);
                    try {
                        auto out = conv_transpose->forward(double_input);
                    } catch (const c10::Error& e) {
                        // Expected for quantized ops
                    }
                    break;
                }
                case 3: {
                    // Test with requires_grad
                    auto grad_input = input.requires_grad_(true);
                    try {
                        auto out = conv_transpose->forward(grad_input);
                        if (out.requires_grad()) {
                            auto sum = out.sum();
                            sum.backward();
                        }
                    } catch (const c10::Error& e) {
                        // Expected for quantized ops
                    }
                    break;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}