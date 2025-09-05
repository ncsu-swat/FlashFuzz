#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeValue(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) return 0;  // Need minimum bytes for parameters
    
    size_t offset = 0;
    
    try {
        // Consume parameters for MaxPool1d
        int64_t kernel_size;
        int64_t stride;
        int64_t padding;
        int64_t dilation;
        bool return_indices;
        bool ceil_mode;
        
        // Consume and constrain parameters
        uint8_t kernel_size_raw, stride_raw, padding_raw, dilation_raw;
        uint8_t flags;
        
        if (!consumeValue(data, size, offset, kernel_size_raw)) return 0;
        if (!consumeValue(data, size, offset, stride_raw)) return 0;
        if (!consumeValue(data, size, offset, padding_raw)) return 0;
        if (!consumeValue(data, size, offset, dilation_raw)) return 0;
        if (!consumeValue(data, size, offset, flags)) return 0;
        
        // Constrain values to reasonable ranges
        kernel_size = (kernel_size_raw % 10) + 1;  // 1-10
        stride = (stride_raw % 10) + 1;  // 1-10
        padding = padding_raw % (kernel_size / 2 + 1);  // 0 to kernel_size/2
        dilation = (dilation_raw % 5) + 1;  // 1-5
        return_indices = flags & 0x01;
        ceil_mode = flags & 0x02;
        
        // Consume tensor dimensions
        uint8_t batch_raw, channels_raw, length_raw;
        uint8_t use_batch;
        
        if (!consumeValue(data, size, offset, batch_raw)) return 0;
        if (!consumeValue(data, size, offset, channels_raw)) return 0;
        if (!consumeValue(data, size, offset, length_raw)) return 0;
        if (!consumeValue(data, size, offset, use_batch)) return 0;
        
        int64_t batch_size = (batch_raw % 8) + 1;  // 1-8
        int64_t channels = (channels_raw % 16) + 1;  // 1-16
        int64_t length = (length_raw % 64) + 1;  // 1-64
        
        // Create input tensor
        torch::Tensor input;
        if (use_batch & 0x01) {
            // 3D input: (N, C, L)
            input = torch::randn({batch_size, channels, length});
        } else {
            // 2D input: (C, L)
            input = torch::randn({channels, length});
        }
        
        // Optionally use different dtypes
        uint8_t dtype_selector;
        if (consumeValue(data, size, offset, dtype_selector)) {
            switch (dtype_selector % 4) {
                case 0:
                    input = input.to(torch::kFloat32);
                    break;
                case 1:
                    input = input.to(torch::kFloat64);
                    break;
                case 2:
                    input = input.to(torch::kFloat16);
                    break;
                case 3:
                    // Keep as is
                    break;
            }
        }
        
        // Fill tensor with fuzzed data if enough bytes remain
        if (offset < size) {
            size_t tensor_bytes = std::min(size - offset, (size_t)(input.numel() * 4));
            if (input.dtype() == torch::kFloat32 && tensor_bytes >= 4) {
                float* data_ptr = input.data_ptr<float>();
                size_t num_floats = tensor_bytes / sizeof(float);
                for (size_t i = 0; i < num_floats && i < input.numel(); ++i) {
                    float val;
                    if (consumeValue(data, size, offset, val)) {
                        // Constrain to reasonable range to avoid NaN/Inf issues
                        if (std::isfinite(val)) {
                            data_ptr[i] = std::clamp(val, -1000.0f, 1000.0f);
                        }
                    }
                }
            }
        }
        
        // Create MaxPool1d module with fuzzed parameters
        torch::nn::MaxPool1dOptions options(kernel_size);
        options.stride(stride);
        options.padding(padding);
        options.dilation(dilation);
        options.ceil_mode(ceil_mode);
        
        torch::nn::MaxPool1d pool(options);
        
        // Apply pooling
        if (return_indices) {
            // Use functional API for return_indices
            auto result = torch::nn::functional::max_pool1d_with_indices(
                input,
                torch::nn::functional::MaxPool1dFuncOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode)
            );
            
            torch::Tensor output = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Verify output shapes
            if (output.dim() > 0 && indices.dim() > 0) {
                // Success - pooling completed
            }
        } else {
            torch::Tensor output = pool->forward(input);
            
            // Verify output shape
            if (output.dim() > 0) {
                // Success - pooling completed
            }
        }
        
        // Test edge cases with requires_grad
        uint8_t test_grad;
        if (consumeValue(data, size, offset, test_grad) && (test_grad & 0x01)) {
            input.requires_grad_(true);
            torch::Tensor output = pool->forward(input);
            if (output.requires_grad()) {
                // Compute backward pass
                torch::Tensor grad_output = torch::ones_like(output);
                output.backward(grad_output);
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid configurations
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