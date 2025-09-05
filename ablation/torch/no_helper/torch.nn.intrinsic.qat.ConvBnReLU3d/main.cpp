#include <torch/torch.h>
#include <torch/nn/intrinsic/qat.h>
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        const uint8_t* data = Data;
        size_t remaining = Size;
        
        // Consume parameters for ConvBnReLU3d
        int64_t in_channels = 1, out_channels = 1;
        int64_t kernel_d = 1, kernel_h = 1, kernel_w = 1;
        int64_t stride_d = 1, stride_h = 1, stride_w = 1;
        int64_t padding_d = 0, padding_h = 0, padding_w = 0;
        int64_t dilation_d = 1, dilation_h = 1, dilation_w = 1;
        int64_t groups = 1;
        bool bias = true;
        bool use_bn = true;
        
        // Parse conv parameters
        uint8_t tmp8;
        if (consumeBytes(data, remaining, tmp8)) {
            in_channels = 1 + (tmp8 % 64);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            out_channels = 1 + (tmp8 % 64);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            kernel_d = 1 + (tmp8 % 7);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            kernel_h = 1 + (tmp8 % 7);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            kernel_w = 1 + (tmp8 % 7);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            stride_d = 1 + (tmp8 % 3);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            stride_h = 1 + (tmp8 % 3);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            stride_w = 1 + (tmp8 % 3);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            padding_d = tmp8 % 3;
        }
        if (consumeBytes(data, remaining, tmp8)) {
            padding_h = tmp8 % 3;
        }
        if (consumeBytes(data, remaining, tmp8)) {
            padding_w = tmp8 % 3;
        }
        if (consumeBytes(data, remaining, tmp8)) {
            dilation_d = 1 + (tmp8 % 2);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            dilation_h = 1 + (tmp8 % 2);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            dilation_w = 1 + (tmp8 % 2);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            groups = 1 + (tmp8 % std::min(in_channels, out_channels));
            // Ensure groups divides both in_channels and out_channels
            while (in_channels % groups != 0 || out_channels % groups != 0) {
                groups = (groups > 1) ? groups - 1 : 1;
            }
        }
        if (consumeBytes(data, remaining, tmp8)) {
            bias = tmp8 & 1;
        }
        
        // BatchNorm parameters
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (consumeBytes(data, remaining, tmp8)) {
            eps = 1e-8 + (tmp8 / 255.0) * 1e-3;
        }
        if (consumeBytes(data, remaining, tmp8)) {
            momentum = (tmp8 / 255.0);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            affine = tmp8 & 1;
        }
        if (consumeBytes(data, remaining, tmp8)) {
            track_running_stats = tmp8 & 1;
        }
        
        // Create ConvBnReLU3d module
        torch::nn::intrinsic::qat::ConvBnReLU3d module(
            torch::nn::Conv3dOptions(in_channels, out_channels, {kernel_d, kernel_h, kernel_w})
                .stride({stride_d, stride_h, stride_w})
                .padding({padding_d, padding_h, padding_w})
                .dilation({dilation_d, dilation_h, dilation_w})
                .groups(groups)
                .bias(bias),
            torch::nn::BatchNorm3dOptions(out_channels)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Generate input tensor dimensions
        int64_t batch_size = 1;
        int64_t depth = 8, height = 8, width = 8;
        
        if (consumeBytes(data, remaining, tmp8)) {
            batch_size = 1 + (tmp8 % 4);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            depth = kernel_d + (tmp8 % 16);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            height = kernel_h + (tmp8 % 16);
        }
        if (consumeBytes(data, remaining, tmp8)) {
            width = kernel_w + (tmp8 % 16);
        }
        
        // Create input tensor
        torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width});
        
        // Fill input with fuzzed data if available
        if (remaining > 0) {
            size_t tensor_size = input.numel() * sizeof(float);
            size_t copy_size = std::min(remaining, tensor_size);
            if (copy_size > 0) {
                std::memcpy(input.data_ptr(), data, copy_size);
                data += copy_size;
                remaining -= copy_size;
            }
        }
        
        // Set module to training or eval mode
        if (consumeBytes(data, remaining, tmp8)) {
            if (tmp8 & 1) {
                module.train();
            } else {
                module.eval();
            }
        }
        
        // Forward pass
        torch::Tensor output = module.forward(input);
        
        // Additional operations to increase coverage
        if (consumeBytes(data, remaining, tmp8)) {
            if (tmp8 & 1) {
                // Backward pass
                torch::Tensor grad_output = torch::randn_like(output);
                output.backward(grad_output);
            }
        }
        
        // Access weight fake quantizer
        auto weight_fake_quant = module.weight_fake_quant();
        if (weight_fake_quant) {
            // Try to use the fake quantizer
            torch::Tensor test_weight = torch::randn({out_channels, in_channels / groups, 
                                                      kernel_d, kernel_h, kernel_w});
            weight_fake_quant->forward(test_weight);
        }
        
        // Try different quantization configurations
        if (consumeBytes(data, remaining, tmp8)) {
            int qscheme_idx = tmp8 % 3;
            if (qscheme_idx == 0) {
                module.qconfig(torch::nn::QConfig());
            } else if (qscheme_idx == 1) {
                module.qconfig(torch::nn::QConfig(
                    torch::nn::MinMaxObserver(),
                    torch::nn::PerChannelMinMaxObserver()
                ));
            }
        }
        
        // Test with zero-dimensional edge cases
        if (consumeBytes(data, remaining, tmp8) && (tmp8 & 1)) {
            try {
                torch::Tensor edge_input = torch::randn({0, in_channels, depth, height, width});
                module.forward(edge_input);
            } catch (...) {
                // Ignore edge case failures
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exceptions
        return -1;
    }
    
    return 0;
}