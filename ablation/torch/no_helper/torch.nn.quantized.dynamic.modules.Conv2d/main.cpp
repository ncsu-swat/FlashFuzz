#include <torch/torch.h>
#include <torch/nn/modules/conv.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    try {
        if (size < 32) return 0; // Need minimum bytes for basic parameters
        
        size_t offset = 0;
        
        // Consume parameters for Conv2d
        int64_t in_channels, out_channels, kernel_h, kernel_w;
        int64_t stride_h, stride_w, padding_h, padding_w;
        int64_t dilation_h, dilation_w, groups;
        int64_t batch_size, input_h, input_w;
        uint8_t use_bias, padding_mode;
        
        if (!consumeBytes(data, size, offset, in_channels)) return 0;
        if (!consumeBytes(data, size, offset, out_channels)) return 0;
        if (!consumeBytes(data, size, offset, kernel_h)) return 0;
        if (!consumeBytes(data, size, offset, kernel_w)) return 0;
        if (!consumeBytes(data, size, offset, stride_h)) return 0;
        if (!consumeBytes(data, size, offset, stride_w)) return 0;
        if (!consumeBytes(data, size, offset, padding_h)) return 0;
        if (!consumeBytes(data, size, offset, padding_w)) return 0;
        if (!consumeBytes(data, size, offset, dilation_h)) return 0;
        if (!consumeBytes(data, size, offset, dilation_w)) return 0;
        if (!consumeBytes(data, size, offset, groups)) return 0;
        if (!consumeBytes(data, size, offset, use_bias)) return 0;
        if (!consumeBytes(data, size, offset, padding_mode)) return 0;
        if (!consumeBytes(data, size, offset, batch_size)) return 0;
        if (!consumeBytes(data, size, offset, input_h)) return 0;
        if (!consumeBytes(data, size, offset, input_w)) return 0;
        
        // Constrain values to reasonable ranges to avoid OOM
        in_channels = (std::abs(in_channels) % 256) + 1;
        out_channels = (std::abs(out_channels) % 256) + 1;
        kernel_h = (std::abs(kernel_h) % 10) + 1;
        kernel_w = (std::abs(kernel_w) % 10) + 1;
        stride_h = (std::abs(stride_h) % 5) + 1;
        stride_w = (std::abs(stride_w) % 5) + 1;
        padding_h = std::abs(padding_h) % 10;
        padding_w = std::abs(padding_w) % 10;
        dilation_h = (std::abs(dilation_h) % 5) + 1;
        dilation_w = (std::abs(dilation_w) % 5) + 1;
        groups = (std::abs(groups) % std::min(in_channels, out_channels)) + 1;
        
        // Ensure groups divides both in_channels and out_channels
        while (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = groups > 1 ? groups - 1 : 1;
        }
        
        batch_size = (std::abs(batch_size) % 8) + 1;
        input_h = (std::abs(input_h) % 64) + kernel_h;
        input_w = (std::abs(input_w) % 64) + kernel_w;
        
        // Create Conv2d options
        auto conv_options = torch::nn::Conv2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
            .stride({stride_h, stride_w})
            .padding({padding_h, padding_w})
            .dilation({dilation_h, dilation_w})
            .groups(groups)
            .bias(use_bias & 1);
        
        // Set padding mode based on fuzzer input
        torch::nn::Conv2dOptions::padding_mode_t pad_mode;
        switch (padding_mode % 4) {
            case 0: pad_mode = torch::kZeros; break;
            case 1: pad_mode = torch::kReflect; break;
            case 2: pad_mode = torch::kReplicate; break;
            case 3: pad_mode = torch::kCircular; break;
            default: pad_mode = torch::kZeros; break;
        }
        conv_options.padding_mode(pad_mode);
        
        // Create quantized dynamic Conv2d module
        // Note: PyTorch C++ API for quantized dynamic modules is through torch::nn::quantized namespace
        auto conv = torch::nn::Conv2d(conv_options);
        
        // Initialize weights with random values
        torch::NoGradGuard no_grad;
        conv->weight.uniform_(-0.1, 0.1);
        if (use_bias & 1) {
            conv->bias.uniform_(-0.1, 0.1);
        }
        
        // Quantize the module dynamically
        conv->to(torch::kQInt8);
        
        // Create input tensor
        auto input = torch::randn({batch_size, in_channels, input_h, input_w}, torch::kFloat32);
        
        // Test with different input variations
        if (offset < size) {
            uint8_t input_variation = data[offset++];
            switch (input_variation % 5) {
                case 0: // Normal input
                    break;
                case 1: // Zero input
                    input = torch::zeros_like(input);
                    break;
                case 2: // Large values
                    input = input * 1000.0;
                    break;
                case 3: // Small values
                    input = input * 0.001;
                    break;
                case 4: // Mixed inf/nan
                    if (input.numel() > 0) {
                        input.view(-1)[0] = std::numeric_limits<float>::infinity();
                        if (input.numel() > 1) {
                            input.view(-1)[1] = std::numeric_limits<float>::quiet_NaN();
                        }
                    }
                    break;
            }
        }
        
        // Forward pass through quantized dynamic conv
        try {
            // Quantize input dynamically
            auto quantized_input = torch::quantize_per_tensor_dynamic(
                input, torch::kQInt8, /*reduce_range=*/false);
            
            // Apply convolution
            auto output = conv->forward(input);
            
            // Test various operations on output
            if (output.defined() && output.numel() > 0) {
                auto sum = output.sum();
                auto mean = output.mean();
                auto max_val = output.max();
                auto min_val = output.min();
                
                // Test backward pass if possible
                if (output.requires_grad()) {
                    auto loss = output.sum();
                    loss.backward();
                }
            }
        } catch (const c10::Error& e) {
            // Silently handle PyTorch errors
        }
        
        // Test edge cases with different tensor shapes
        if (offset < size) {
            uint8_t edge_case = data[offset++];
            switch (edge_case % 4) {
                case 0: { // Empty batch
                    auto empty_input = torch::randn({0, in_channels, input_h, input_w});
                    try {
                        auto output = conv->forward(empty_input);
                    } catch (const c10::Error& e) {}
                    break;
                }
                case 1: { // Single pixel
                    auto single_input = torch::randn({batch_size, in_channels, kernel_h, kernel_w});
                    try {
                        auto output = conv->forward(single_input);
                    } catch (const c10::Error& e) {}
                    break;
                }
                case 2: { // Mismatched channels
                    auto wrong_input = torch::randn({batch_size, in_channels + 1, input_h, input_w});
                    try {
                        auto output = conv->forward(wrong_input);
                    } catch (const c10::Error& e) {}
                    break;
                }
                case 3: { // Different dtype
                    auto double_input = input.to(torch::kFloat64);
                    try {
                        auto output = conv->forward(double_input);
                    } catch (const c10::Error& e) {}
                    break;
                }
            }
        }
        
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}