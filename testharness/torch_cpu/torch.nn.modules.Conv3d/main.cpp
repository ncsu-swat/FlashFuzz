#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

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
        // Early exit for very small inputs
        if (Size < 20) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse parameters first to know in_channels before creating input
        // Parse out_channels
        int64_t out_channels = 1;
        if (offset + sizeof(uint8_t) <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            offset++;
        }
        
        // Parse in_channels
        int64_t in_channels = 1;
        if (offset + sizeof(uint8_t) <= Size) {
            in_channels = (Data[offset] % 8) + 1;
            offset++;
        }
        
        // Parse kernel_size (use smaller values for efficiency)
        int64_t kernel_d = 1, kernel_h = 1, kernel_w = 1;
        if (offset + 3 <= Size) {
            kernel_d = (Data[offset] % 3) + 1;
            kernel_h = (Data[offset + 1] % 3) + 1;
            kernel_w = (Data[offset + 2] % 3) + 1;
            offset += 3;
        }
        
        // Parse stride
        int64_t stride_d = 1, stride_h = 1, stride_w = 1;
        if (offset + 3 <= Size) {
            stride_d = (Data[offset] % 2) + 1;
            stride_h = (Data[offset + 1] % 2) + 1;
            stride_w = (Data[offset + 2] % 2) + 1;
            offset += 3;
        }
        
        // Parse padding
        int64_t padding_d = 0, padding_h = 0, padding_w = 0;
        if (offset + 3 <= Size) {
            padding_d = Data[offset] % 2;
            padding_h = Data[offset + 1] % 2;
            padding_w = Data[offset + 2] % 2;
            offset += 3;
        }
        
        // Parse dilation
        int64_t dilation_d = 1, dilation_h = 1, dilation_w = 1;
        if (offset + 3 <= Size) {
            dilation_d = (Data[offset] % 2) + 1;
            dilation_h = (Data[offset + 1] % 2) + 1;
            dilation_w = (Data[offset + 2] % 2) + 1;
            offset += 3;
        }
        
        // Parse groups - must divide in_channels evenly
        int64_t groups = 1;
        if (offset + sizeof(uint8_t) <= Size) {
            // Find valid divisor of in_channels
            int64_t g = (Data[offset] % in_channels) + 1;
            while (in_channels % g != 0 && g > 1) {
                g--;
            }
            groups = g;
            offset++;
        }
        
        // Parse bias flag
        bool use_bias = true;
        if (offset < Size) {
            use_bias = Data[offset] & 1;
            offset++;
        }
        
        // Parse spatial dimensions for input tensor
        int64_t batch_size = 1;
        int64_t depth = 4, height = 4, width = 4;
        if (offset + 4 <= Size) {
            batch_size = (Data[offset] % 4) + 1;
            depth = (Data[offset + 1] % 8) + kernel_d;  // Ensure >= kernel size
            height = (Data[offset + 2] % 8) + kernel_h;
            width = (Data[offset + 3] % 8) + kernel_w;
            offset += 4;
        }
        
        // Create input tensor with correct shape (N, C, D, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width});
        
        // Use remaining data to modify input values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining / sizeof(float), 
                                           static_cast<size_t>(input.numel()));
            if (num_elements > 0) {
                auto input_accessor = input.accessor<float, 5>();
                size_t idx = 0;
                for (int64_t n = 0; n < batch_size && idx < num_elements; n++) {
                    for (int64_t c = 0; c < in_channels && idx < num_elements; c++) {
                        for (int64_t d = 0; d < depth && idx < num_elements; d++) {
                            for (int64_t h = 0; h < height && idx < num_elements; h++) {
                                for (int64_t w = 0; w < width && idx < num_elements; w++) {
                                    float val;
                                    std::memcpy(&val, Data + offset + idx * sizeof(float), sizeof(float));
                                    if (std::isfinite(val)) {
                                        input_accessor[n][c][d][h][w] = val;
                                    }
                                    idx++;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Create Conv3d module with parsed options
        torch::nn::Conv3dOptions options(in_channels, out_channels, 
                                         torch::ExpandingArray<3>({kernel_d, kernel_h, kernel_w}));
        options.stride(torch::ExpandingArray<3>({stride_d, stride_h, stride_w}))
               .padding(torch::ExpandingArray<3>({padding_d, padding_h, padding_w}))
               .dilation(torch::ExpandingArray<3>({dilation_d, dilation_h, dilation_w}))
               .groups(groups)
               .bias(use_bias);
        
        torch::nn::Conv3d conv(options);
        
        // Apply Conv3d to input tensor
        torch::Tensor output = conv->forward(input);
        
        // Additional coverage: test with different padding modes
        // Default is "zeros", also test "replicate" and "circular"
        if (offset < Size) {
            int mode_select = Data[offset % Size] % 3;
            
            torch::nn::Conv3dOptions options2(in_channels, out_channels,
                                              torch::ExpandingArray<3>({kernel_d, kernel_h, kernel_w}));
            options2.stride(torch::ExpandingArray<3>({stride_d, stride_h, stride_w}))
                    .padding(torch::ExpandingArray<3>({padding_d, padding_h, padding_w}))
                    .dilation(torch::ExpandingArray<3>({dilation_d, dilation_h, dilation_w}))
                    .groups(groups)
                    .bias(use_bias);
            
            if (mode_select == 1) {
                options2.padding_mode(torch::kReplicate);
            } else if (mode_select == 2) {
                options2.padding_mode(torch::kCircular);
            }
            // mode_select == 0 keeps default (zeros)
            
            try {
                torch::nn::Conv3d conv2(options2);
                torch::Tensor output2 = conv2->forward(input);
            } catch (const std::exception&) {
                // Some padding modes may not work with all configurations
            }
        }
        
        // Test training vs eval mode
        conv->train();
        torch::Tensor train_output = conv->forward(input);
        
        conv->eval();
        torch::Tensor eval_output = conv->forward(input);
        
        // Test with gradient computation
        torch::Tensor grad_input = input.clone();
        grad_input.set_requires_grad(true);
        torch::Tensor grad_output = conv->forward(grad_input);
        
        try {
            grad_output.sum().backward();
        } catch (const std::exception&) {
            // Gradient computation may fail in some edge cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}