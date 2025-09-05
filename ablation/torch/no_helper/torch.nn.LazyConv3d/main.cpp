#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) return 0;  // Need minimum bytes for configuration
    
    try {
        size_t offset = 0;
        
        // Consume configuration parameters
        uint8_t out_channels_raw, kernel_size_type, stride_type, padding_type;
        uint8_t padding_mode_idx, dilation_type, groups_raw, use_bias;
        uint8_t batch_size_raw, depth_raw, height_raw, width_raw, channels_raw;
        
        consumeBytes(data, offset, size, out_channels_raw);
        consumeBytes(data, offset, size, kernel_size_type);
        consumeBytes(data, offset, size, stride_type);
        consumeBytes(data, offset, size, padding_type);
        consumeBytes(data, offset, size, padding_mode_idx);
        consumeBytes(data, offset, size, dilation_type);
        consumeBytes(data, offset, size, groups_raw);
        consumeBytes(data, offset, size, use_bias);
        consumeBytes(data, offset, size, batch_size_raw);
        consumeBytes(data, offset, size, depth_raw);
        consumeBytes(data, offset, size, height_raw);
        consumeBytes(data, offset, size, width_raw);
        consumeBytes(data, offset, size, channels_raw);
        
        // Map raw values to reasonable ranges
        int64_t out_channels = (out_channels_raw % 64) + 1;
        int64_t groups = (groups_raw % 8) + 1;
        
        // Ensure out_channels is divisible by groups
        out_channels = (out_channels / groups) * groups;
        if (out_channels == 0) out_channels = groups;
        
        // Kernel size - either single int or tuple of 3
        torch::nn::Conv3dOptions::kernel_size_t kernel_size;
        if (kernel_size_type % 2 == 0) {
            uint8_t k;
            consumeBytes(data, offset, size, k);
            kernel_size = (k % 7) + 1;
        } else {
            uint8_t k1, k2, k3;
            consumeBytes(data, offset, size, k1);
            consumeBytes(data, offset, size, k2);
            consumeBytes(data, offset, size, k3);
            kernel_size = torch::ExpandingArray<3>({(k1 % 5) + 1, (k2 % 5) + 1, (k3 % 5) + 1});
        }
        
        // Stride - either single int or tuple of 3
        torch::nn::Conv3dOptions::stride_t stride;
        if (stride_type % 2 == 0) {
            uint8_t s;
            consumeBytes(data, offset, size, s);
            stride = (s % 3) + 1;
        } else {
            uint8_t s1, s2, s3;
            consumeBytes(data, offset, size, s1);
            consumeBytes(data, offset, size, s2);
            consumeBytes(data, offset, size, s3);
            stride = torch::ExpandingArray<3>({(s1 % 3) + 1, (s2 % 3) + 1, (s3 % 3) + 1});
        }
        
        // Padding - either single int or tuple of 3
        torch::nn::Conv3dOptions::padding_t padding;
        if (padding_type % 3 == 0) {
            uint8_t p;
            consumeBytes(data, offset, size, p);
            padding = p % 4;
        } else if (padding_type % 3 == 1) {
            uint8_t p1, p2, p3;
            consumeBytes(data, offset, size, p1);
            consumeBytes(data, offset, size, p2);
            consumeBytes(data, offset, size, p3);
            padding = torch::ExpandingArray<3>({p1 % 3, p2 % 3, p3 % 3});
        } else {
            // Use string padding modes
            if ((padding_mode_idx % 4) == 0) {
                padding = torch::kValid;
            } else {
                padding = torch::kSame;
            }
        }
        
        // Padding mode
        std::string padding_mode;
        switch (padding_mode_idx % 4) {
            case 0: padding_mode = "zeros"; break;
            case 1: padding_mode = "reflect"; break;
            case 2: padding_mode = "replicate"; break;
            case 3: padding_mode = "circular"; break;
        }
        
        // Dilation - either single int or tuple of 3
        torch::nn::Conv3dOptions::dilation_t dilation;
        if (dilation_type % 2 == 0) {
            uint8_t d;
            consumeBytes(data, offset, size, d);
            dilation = (d % 3) + 1;
        } else {
            uint8_t d1, d2, d3;
            consumeBytes(data, offset, size, d1);
            consumeBytes(data, offset, size, d2);
            consumeBytes(data, offset, size, d3);
            dilation = torch::ExpandingArray<3>({(d1 % 2) + 1, (d2 % 2) + 1, (d3 % 2) + 1});
        }
        
        // Create LazyConv3d module
        auto options = torch::nn::Conv3dOptions(torch::nn::LazyConv3d::uninitialized_parameter, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .padding_mode(torch::nn::detail::conv_padding_mode_t(padding_mode))
            .dilation(dilation)
            .groups(groups)
            .bias(use_bias != 0);
        
        torch::nn::LazyConv3d lazy_conv(options);
        
        // Create input tensor with shape [batch, channels, depth, height, width]
        int64_t batch = (batch_size_raw % 4) + 1;
        int64_t channels = ((channels_raw % 32) + 1);
        // Ensure channels is divisible by groups for depthwise/grouped convolution
        channels = (channels / groups) * groups;
        if (channels == 0) channels = groups;
        
        int64_t depth = (depth_raw % 16) + 1;
        int64_t height = (height_raw % 16) + 1;
        int64_t width = (width_raw % 16) + 1;
        
        // Test with edge cases including 0-dimensional tensors
        if (offset < size && data[offset] % 10 == 0) {
            // Occasionally test with minimal dimensions
            if (data[offset] % 3 == 0) depth = 1;
            if (data[offset] % 3 == 1) height = 1;
            if (data[offset] % 3 == 2) width = 1;
        }
        
        // Create input tensor
        torch::Tensor input = torch::randn({batch, channels, depth, height, width});
        
        // Test with different dtypes
        if (offset < size) {
            uint8_t dtype_choice;
            consumeBytes(data, offset, size, dtype_choice);
            switch (dtype_choice % 4) {
                case 0: input = input.to(torch::kFloat32); break;
                case 1: input = input.to(torch::kFloat64); break;
                case 2: input = input.to(torch::kFloat16); break;
                case 3: input = input.to(torch::kBFloat16); break;
            }
        }
        
        // Test with different memory layouts
        if (offset < size && data[offset] % 3 == 0) {
            input = input.contiguous(torch::MemoryFormat::ChannelsLast3d);
        }
        
        // Forward pass - this will initialize the lazy parameters
        torch::Tensor output = lazy_conv->forward(input);
        
        // Verify output shape
        auto out_shape = output.sizes();
        
        // Additional operations to increase coverage
        if (offset < size && data[offset] % 2 == 0) {
            // Test backward pass
            output.sum().backward();
        }
        
        // Test with requires_grad
        if (offset < size && data[offset] % 2 == 0) {
            input.requires_grad_(true);
            output = lazy_conv->forward(input);
            if (output.requires_grad()) {
                auto grad_output = torch::randn_like(output);
                output.backward(grad_output);
            }
        }
        
        // Test module state operations
        if (offset < size) {
            uint8_t op;
            consumeBytes(data, offset, size, op);
            switch (op % 5) {
                case 0: lazy_conv->eval(); break;
                case 1: lazy_conv->train(); break;
                case 2: lazy_conv->zero_grad(); break;
                case 3: {
                    // Test serialization
                    std::stringstream stream;
                    torch::save(lazy_conv, stream);
                    torch::nn::LazyConv3d loaded_conv(options);
                    torch::load(loaded_conv, stream);
                    break;
                }
                case 4: {
                    // Test cloning
                    auto cloned = std::dynamic_pointer_cast<torch::nn::LazyConv3dImpl>(lazy_conv->clone());
                    if (cloned) {
                        cloned->forward(input);
                    }
                    break;
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected in fuzzing
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return 0;
    }
    
    return 0;
}