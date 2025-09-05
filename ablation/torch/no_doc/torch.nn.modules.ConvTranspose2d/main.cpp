#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume bytes and create module parameters
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    // Clamp to reasonable range
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    if (Size < 10) {
        return 0; // Need minimum bytes for basic parameters
    }

    try
    {
        size_t offset = 0;

        // Parse ConvTranspose2d parameters from fuzzer input
        int64_t in_channels = consumeValue<int64_t>(Data, offset, Size, 1, 512);
        int64_t out_channels = consumeValue<int64_t>(Data, offset, Size, 1, 512);
        
        // Kernel size - can be single value or tuple
        bool use_tuple_kernel = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        int64_t kernel_h = consumeValue<int64_t>(Data, offset, Size, 1, 11);
        int64_t kernel_w = use_tuple_kernel ? consumeValue<int64_t>(Data, offset, Size, 1, 11) : kernel_h;
        
        // Stride
        bool use_tuple_stride = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        int64_t stride_h = consumeValue<int64_t>(Data, offset, Size, 1, 5);
        int64_t stride_w = use_tuple_stride ? consumeValue<int64_t>(Data, offset, Size, 1, 5) : stride_h;
        
        // Padding
        bool use_tuple_padding = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        int64_t padding_h = consumeValue<int64_t>(Data, offset, Size, 0, 5);
        int64_t padding_w = use_tuple_padding ? consumeValue<int64_t>(Data, offset, Size, 0, 5) : padding_h;
        
        // Output padding
        bool use_tuple_output_padding = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        int64_t output_padding_h = consumeValue<int64_t>(Data, offset, Size, 0, stride_h - 1);
        int64_t output_padding_w = use_tuple_output_padding ? 
            consumeValue<int64_t>(Data, offset, Size, 0, stride_w - 1) : output_padding_h;
        
        // Groups
        int64_t groups = consumeValue<int64_t>(Data, offset, Size, 1, std::min(in_channels, out_channels));
        // Ensure divisibility
        while (groups > 1 && (in_channels % groups != 0 || out_channels % groups != 0)) {
            groups--;
        }
        
        // Bias
        bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Dilation
        bool use_tuple_dilation = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        int64_t dilation_h = consumeValue<int64_t>(Data, offset, Size, 1, 3);
        int64_t dilation_w = use_tuple_dilation ? consumeValue<int64_t>(Data, offset, Size, 1, 3) : dilation_h;
        
        // Padding mode - ConvTranspose2d only supports 'zeros'
        torch::nn::detail::conv_padding_mode_t padding_mode = torch::kZeros;

        // Create ConvTranspose2d module with options
        auto options = torch::nn::ConvTranspose2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
            .stride({stride_h, stride_w})
            .padding({padding_h, padding_w})
            .output_padding({output_padding_h, output_padding_w})
            .groups(groups)
            .bias(bias)
            .dilation({dilation_h, dilation_w})
            .padding_mode(padding_mode);

        torch::nn::ConvTranspose2d conv_transpose(options);

        // Create input tensor from remaining fuzzer data
        torch::Tensor input;
        if (offset < Size) {
            try {
                input = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (...) {
                // If tensor creation fails, create a default one
                input = torch::randn({1, in_channels, 4, 4});
            }
        } else {
            // Default input if no data left
            input = torch::randn({1, in_channels, 4, 4});
        }

        // Ensure input has correct number of dimensions and channels
        if (input.dim() < 4) {
            // Reshape to 4D if needed
            std::vector<int64_t> new_shape(4, 1);
            auto flat_size = input.numel();
            new_shape[1] = in_channels;
            
            // Calculate spatial dimensions
            int64_t spatial_size = flat_size / in_channels;
            if (spatial_size <= 0) spatial_size = 1;
            
            int64_t h = static_cast<int64_t>(std::sqrt(spatial_size));
            if (h <= 0) h = 1;
            int64_t w = spatial_size / h;
            if (w <= 0) w = 1;
            
            new_shape[2] = h;
            new_shape[3] = w;
            
            // Only reshape if total elements allows it
            if (flat_size >= in_channels * h * w) {
                input = input.view(new_shape);
            } else {
                input = torch::randn({1, in_channels, 4, 4}, input.options());
            }
        } else if (input.dim() > 4) {
            // Flatten extra dimensions
            input = input.flatten(0, input.dim() - 4);
        }

        // Adjust channel dimension if needed
        if (input.size(1) != in_channels) {
            if (input.size(1) < in_channels) {
                // Pad channels
                auto padding_tensor = torch::zeros({input.size(0), in_channels - input.size(1), 
                                                   input.size(2), input.size(3)}, input.options());
                input = torch::cat({input, padding_tensor}, 1);
            } else {
                // Truncate channels
                input = input.slice(1, 0, in_channels);
            }
        }

        // Test different scenarios
        
        // 1. Forward pass
        torch::Tensor output = conv_transpose->forward(input);
        
        // 2. Try with different input sizes (edge cases)
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Very small spatial dimensions
            auto small_input = torch::randn({2, in_channels, 1, 1}, input.options());
            auto small_output = conv_transpose->forward(small_input);
        }
        
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Larger batch size
            auto batch_input = torch::randn({8, in_channels, 3, 3}, input.options());
            auto batch_output = conv_transpose->forward(batch_input);
        }

        // 3. Test gradient computation if possible
        if (offset < Size && Data[offset++] % 2 == 0) {
            input.requires_grad_(true);
            auto out_grad = conv_transpose->forward(input);
            if (out_grad.numel() > 0) {
                auto loss = out_grad.sum();
                loss.backward();
            }
        }

        // 4. Test with eval mode
        conv_transpose->eval();
        auto eval_output = conv_transpose->forward(input);
        
        // 5. Test with train mode
        conv_transpose->train();
        auto train_output = conv_transpose->forward(input);

        // 6. Access and manipulate parameters
        for (auto& param : conv_transpose->parameters()) {
            if (offset < Size && Data[offset++] % 4 == 0) {
                // Occasionally zero out parameters
                param.zero_();
            } else if (offset < Size && Data[offset++] % 4 == 1) {
                // Or set to ones
                param.fill_(1.0);
            }
        }
        
        // 7. Test after parameter manipulation
        auto modified_output = conv_transpose->forward(input);

        // 8. Test state_dict operations
        auto state = conv_transpose->state_dict();
        
        // 9. Test with different tensor properties
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Try with non-contiguous input
            if (input.size(2) > 1 && input.size(3) > 1) {
                auto transposed = input.transpose(2, 3);
                auto trans_output = conv_transpose->forward(transposed);
            }
        }

        // 10. Edge case: single pixel input
        auto single_pixel = torch::randn({1, in_channels, 1, 1}, input.options());
        auto single_output = conv_transpose->forward(single_pixel);

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}