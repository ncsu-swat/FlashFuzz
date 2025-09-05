#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    
    // Ensure value is in range [min_val, max_val]
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 20) {
            // Need minimum bytes for configuration
            return 0;
        }

        size_t offset = 0;
        
        // Parse LazyConv3d configuration
        int64_t out_channels = consumeValue<int64_t>(Data, offset, Size, 1, 256);
        
        // Parse kernel size (can be single int or tuple of 3)
        bool use_tuple_kernel = (offset < Size) ? (Data[offset++] % 2) : false;
        std::vector<int64_t> kernel_size;
        if (use_tuple_kernel) {
            for (int i = 0; i < 3; i++) {
                kernel_size.push_back(consumeValue<int64_t>(Data, offset, Size, 1, 7));
            }
        } else {
            int64_t k = consumeValue<int64_t>(Data, offset, Size, 1, 7);
            kernel_size = {k, k, k};
        }
        
        // Parse stride (can be single int or tuple of 3)
        bool use_tuple_stride = (offset < Size) ? (Data[offset++] % 2) : false;
        std::vector<int64_t> stride;
        if (use_tuple_stride) {
            for (int i = 0; i < 3; i++) {
                stride.push_back(consumeValue<int64_t>(Data, offset, Size, 1, 3));
            }
        } else {
            int64_t s = consumeValue<int64_t>(Data, offset, Size, 1, 3);
            stride = {s, s, s};
        }
        
        // Parse padding (can be single int or tuple of 3)
        bool use_tuple_padding = (offset < Size) ? (Data[offset++] % 2) : false;
        std::vector<int64_t> padding;
        if (use_tuple_padding) {
            for (int i = 0; i < 3; i++) {
                padding.push_back(consumeValue<int64_t>(Data, offset, Size, 0, 3));
            }
        } else {
            int64_t p = consumeValue<int64_t>(Data, offset, Size, 0, 3);
            padding = {p, p, p};
        }
        
        // Parse dilation (can be single int or tuple of 3)
        bool use_tuple_dilation = (offset < Size) ? (Data[offset++] % 2) : false;
        std::vector<int64_t> dilation;
        if (use_tuple_dilation) {
            for (int i = 0; i < 3; i++) {
                dilation.push_back(consumeValue<int64_t>(Data, offset, Size, 1, 3));
            }
        } else {
            int64_t d = consumeValue<int64_t>(Data, offset, Size, 1, 3);
            dilation = {d, d, d};
        }
        
        // Parse groups
        int64_t groups = consumeValue<int64_t>(Data, offset, Size, 1, out_channels);
        
        // Parse bias
        bool bias = (offset < Size) ? (Data[offset++] % 2) : true;
        
        // Parse padding mode (0: zeros, 1: reflect, 2: replicate, 3: circular)
        std::string padding_mode = "zeros";
        if (offset < Size) {
            uint8_t mode = Data[offset++] % 4;
            switch(mode) {
                case 0: padding_mode = "zeros"; break;
                case 1: padding_mode = "reflect"; break;
                case 2: padding_mode = "replicate"; break;
                case 3: padding_mode = "circular"; break;
            }
        }
        
        // Create LazyConv3d module
        torch::nn::LazyConv3dOptions options(out_channels);
        
        // Set kernel size
        if (kernel_size.size() == 3) {
            options.kernel_size(torch::ExpandingArray3<int64_t>({kernel_size[0], kernel_size[1], kernel_size[2]}));
        }
        
        // Set stride
        if (stride.size() == 3) {
            options.stride(torch::ExpandingArray3<int64_t>({stride[0], stride[1], stride[2]}));
        }
        
        // Set padding
        if (padding.size() == 3) {
            options.padding(torch::ExpandingArray3<int64_t>({padding[0], padding[1], padding[2]}));
        }
        
        // Set dilation
        if (dilation.size() == 3) {
            options.dilation(torch::ExpandingArray3<int64_t>({dilation[0], dilation[1], dilation[2]}));
        }
        
        // Set other options
        options.groups(groups);
        options.bias(bias);
        options.padding_mode(torch::kZeros); // Use enum directly for now
        
        auto lazy_conv3d = torch::nn::LazyConv3d(options);
        
        // Create input tensor
        // LazyConv3d expects 5D input: (batch, channels, depth, height, width)
        torch::Tensor input;
        
        // Try to create tensor from remaining data
        if (offset < Size) {
            try {
                input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure input is 5D
                if (input.dim() != 5) {
                    // Reshape or create new 5D tensor
                    int64_t batch = 1 + (offset < Size ? Data[offset++] % 4 : 0);
                    int64_t channels = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                    int64_t depth = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                    int64_t height = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                    int64_t width = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                    
                    input = torch::randn({batch, channels, depth, height, width});
                }
            } catch (...) {
                // Fallback to random tensor
                int64_t batch = 1 + (offset < Size ? Data[offset++] % 4 : 0);
                int64_t channels = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                int64_t depth = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                int64_t height = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                int64_t width = 1 + (offset < Size ? Data[offset++] % 16 : 0);
                
                input = torch::randn({batch, channels, depth, height, width});
            }
        } else {
            // Create default input
            input = torch::randn({1, 3, 8, 8, 8});
        }
        
        // Ensure input requires grad for testing backward pass
        input.requires_grad_(true);
        
        // Forward pass
        torch::Tensor output = lazy_conv3d->forward(input);
        
        // Test various operations on output
        if (output.numel() > 0) {
            // Test backward pass
            if (output.requires_grad()) {
                torch::Tensor grad_output = torch::ones_like(output);
                output.backward(grad_output);
            }
            
            // Test some tensor operations
            auto sum = output.sum();
            auto mean = output.mean();
            auto max_val = output.max();
            auto min_val = output.min();
            
            // Test reshape operations
            if (output.numel() > 1) {
                auto flattened = output.flatten();
                auto reshaped = output.reshape({-1});
            }
            
            // Test different forward passes with different input sizes
            // This tests the lazy initialization
            if (offset + 10 < Size) {
                int64_t new_channels = 1 + Data[offset++] % 32;
                int64_t new_depth = 1 + Data[offset++] % 16;
                int64_t new_height = 1 + Data[offset++] % 16;
                int64_t new_width = 1 + Data[offset++] % 16;
                
                torch::Tensor input2 = torch::randn({2, new_channels, new_depth, new_height, new_width});
                
                try {
                    // This might fail if already initialized with different channel size
                    torch::Tensor output2 = lazy_conv3d->forward(input2);
                } catch (const c10::Error& e) {
                    // Expected for some inputs after initialization
                }
            }
        }
        
        // Test module methods
        lazy_conv3d->zero_grad();
        auto params = lazy_conv3d->parameters();
        auto named_params = lazy_conv3d->named_parameters();
        
        // Test state dict operations
        auto state_dict = lazy_conv3d->state_dict();
        
        // Test eval/train mode
        lazy_conv3d->eval();
        torch::Tensor eval_output = lazy_conv3d->forward(input);
        
        lazy_conv3d->train();
        torch::Tensor train_output = lazy_conv3d->forward(input);
        
        // Test cloning
        auto cloned = lazy_conv3d->clone();
        
        // Test to() operations for device/dtype conversion
        lazy_conv3d->to(torch::kFloat64);
        lazy_conv3d->to(torch::kFloat32);
        
        // Test pretty print
        std::stringstream ss;
        ss << *lazy_conv3d;
        
    }
    catch (const c10::Error& e)
    {
        // PyTorch errors are expected for some inputs
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}