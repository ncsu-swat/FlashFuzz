#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeValue(const uint8_t*& data, size_t& remaining, T& value) {
    if (remaining < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    remaining -= sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) return 0; // Need minimum bytes for parameters
    
    const uint8_t* ptr = data;
    size_t remaining = size;
    
    try {
        // Consume parameters for ConvTranspose2d
        uint8_t in_channels_raw, out_channels_raw;
        uint8_t kernel_h, kernel_w;
        uint8_t stride_h, stride_w;
        uint8_t padding_h, padding_w;
        uint8_t output_padding_h, output_padding_w;
        uint8_t dilation_h, dilation_w;
        uint8_t groups_raw;
        uint8_t use_bias;
        uint8_t batch_size_raw, input_h, input_w;
        uint8_t dtype_selector;
        
        if (!consumeValue(ptr, remaining, in_channels_raw)) return 0;
        if (!consumeValue(ptr, remaining, out_channels_raw)) return 0;
        if (!consumeValue(ptr, remaining, kernel_h)) return 0;
        if (!consumeValue(ptr, remaining, kernel_w)) return 0;
        if (!consumeValue(ptr, remaining, stride_h)) return 0;
        if (!consumeValue(ptr, remaining, stride_w)) return 0;
        if (!consumeValue(ptr, remaining, padding_h)) return 0;
        if (!consumeValue(ptr, remaining, padding_w)) return 0;
        if (!consumeValue(ptr, remaining, output_padding_h)) return 0;
        if (!consumeValue(ptr, remaining, output_padding_w)) return 0;
        if (!consumeValue(ptr, remaining, dilation_h)) return 0;
        if (!consumeValue(ptr, remaining, dilation_w)) return 0;
        if (!consumeValue(ptr, remaining, groups_raw)) return 0;
        if (!consumeValue(ptr, remaining, use_bias)) return 0;
        if (!consumeValue(ptr, remaining, batch_size_raw)) return 0;
        if (!consumeValue(ptr, remaining, input_h)) return 0;
        if (!consumeValue(ptr, remaining, input_w)) return 0;
        if (!consumeValue(ptr, remaining, dtype_selector)) return 0;
        
        // Convert raw values to reasonable ranges
        int64_t in_channels = (in_channels_raw % 64) + 1;  // 1-64
        int64_t out_channels = (out_channels_raw % 64) + 1; // 1-64
        int64_t kernel_size_h = (kernel_h % 7) + 1; // 1-7
        int64_t kernel_size_w = (kernel_w % 7) + 1; // 1-7
        int64_t stride_val_h = (stride_h % 4) + 1; // 1-4
        int64_t stride_val_w = (stride_w % 4) + 1; // 1-4
        int64_t padding_val_h = padding_h % 5; // 0-4
        int64_t padding_val_w = padding_w % 5; // 0-4
        int64_t output_padding_val_h = output_padding_h % 3; // 0-2
        int64_t output_padding_val_w = output_padding_w % 3; // 0-2
        int64_t dilation_val_h = (dilation_h % 3) + 1; // 1-3
        int64_t dilation_val_w = (dilation_w % 3) + 1; // 1-3
        int64_t groups = (groups_raw % std::min(in_channels, out_channels)) + 1;
        
        // Ensure groups divides both in_channels and out_channels
        while (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = (groups % std::min(in_channels, out_channels)) + 1;
            if (groups == 1) break;
        }
        
        bool bias = use_bias & 1;
        int64_t batch_size = (batch_size_raw % 8) + 1; // 1-8
        int64_t height = (input_h % 32) + 1; // 1-32
        int64_t width = (input_w % 32) + 1; // 1-32
        
        // Ensure output_padding is less than stride
        output_padding_val_h = std::min(output_padding_val_h, stride_val_h - 1);
        output_padding_val_w = std::min(output_padding_val_w, stride_val_w - 1);
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector % 3) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32;
        }
        
        // Create ConvTranspose2d module
        torch::nn::ConvTranspose2dOptions options(in_channels, out_channels, 
            torch::ExpandingArray<2>({kernel_size_h, kernel_size_w}));
        options.stride(torch::ExpandingArray<2>({stride_val_h, stride_val_w}));
        options.padding(torch::ExpandingArray<2>({padding_val_h, padding_val_w}));
        options.output_padding(torch::ExpandingArray<2>({output_padding_val_h, output_padding_val_w}));
        options.dilation(torch::ExpandingArray<2>({dilation_val_h, dilation_val_w}));
        options.groups(groups);
        options.bias(bias);
        
        torch::nn::ConvTranspose2d conv_transpose(options);
        
        // Create input tensor with remaining bytes as data
        torch::Tensor input;
        
        // Try different input creation strategies
        uint8_t tensor_strategy = 0;
        if (remaining > 0) {
            tensor_strategy = ptr[0] % 4;
            ptr++; remaining--;
        }
        
        switch (tensor_strategy) {
            case 0: // Random tensor
                input = torch::randn({batch_size, in_channels, height, width}, 
                                    torch::TensorOptions().dtype(dtype));
                break;
            case 1: // Zeros
                input = torch::zeros({batch_size, in_channels, height, width},
                                    torch::TensorOptions().dtype(dtype));
                break;
            case 2: // Ones
                input = torch::ones({batch_size, in_channels, height, width},
                                   torch::TensorOptions().dtype(dtype));
                break;
            case 3: // From remaining data
                {
                    size_t tensor_size = batch_size * in_channels * height * width;
                    std::vector<float> values(tensor_size);
                    for (size_t i = 0; i < tensor_size && remaining > 0; i++) {
                        values[i] = (ptr[0] - 128.0f) / 128.0f; // Normalize to [-1, 1]
                        ptr++; remaining--;
                    }
                    input = torch::from_blob(values.data(), {batch_size, in_channels, height, width},
                                            torch::kFloat32).to(dtype);
                }
                break;
        }
        
        // Test with batch dimension
        torch::Tensor output = conv_transpose->forward(input);
        
        // Test without batch dimension if possible
        if (batch_size == 1 && remaining > 0 && ptr[0] % 2 == 0) {
            torch::Tensor squeezed_input = input.squeeze(0);
            torch::Tensor squeezed_output = conv_transpose->forward(squeezed_input);
        }
        
        // Test gradient computation if there's remaining data
        if (remaining > 0 && ptr[0] % 2 == 0) {
            input.requires_grad_(true);
            output = conv_transpose->forward(input);
            torch::Tensor loss = output.sum();
            loss.backward();
        }
        
        // Test with different output_size specification
        if (remaining > 1) {
            uint8_t target_h = ptr[0] % 64 + 1;
            uint8_t target_w = ptr[1] % 64 + 1;
            
            // Calculate valid output size range
            int64_t min_h = (height - 1) * stride_val_h - 2 * padding_val_h + 
                           dilation_val_h * (kernel_size_h - 1) + output_padding_val_h + 1;
            int64_t max_h = min_h + stride_val_h - 1;
            int64_t min_w = (width - 1) * stride_val_w - 2 * padding_val_w + 
                           dilation_val_w * (kernel_size_w - 1) + output_padding_val_w + 1;
            int64_t max_w = min_w + stride_val_w - 1;
            
            int64_t out_h = min_h + (target_h % std::max(int64_t(1), max_h - min_h + 1));
            int64_t out_w = min_w + (target_w % std::max(int64_t(1), max_w - min_w + 1));
            
            // Forward with output_size
            std::vector<int64_t> output_size = {out_h, out_w};
            output = conv_transpose->forward(input, output_size);
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid configurations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exception
        return -1;
    }
    
    return 0;
}