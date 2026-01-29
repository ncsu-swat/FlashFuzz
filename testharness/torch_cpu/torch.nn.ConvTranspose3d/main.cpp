#include "fuzzer_utils.h"
#include <iostream>

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
        // Need at least some data to proceed
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Parse parameters for ConvTranspose3d from data first
        int64_t in_channels = (Data[offset++] % 4) + 1;   // 1-4 channels
        int64_t out_channels = (Data[offset++] % 4) + 1;  // 1-4 channels
        int64_t kernel_size = (Data[offset++] % 3) + 1;   // 1-3 kernel size
        int64_t stride = (Data[offset++] % 3) + 1;        // 1-3 stride
        int64_t padding = Data[offset++] % 3;             // 0-2 padding
        int64_t output_padding_val = Data[offset++] % stride; // must be < stride
        int64_t groups = 1;
        int64_t groups_selector = Data[offset++] % 4;     // 0-3
        bool use_bias = Data[offset++] % 2 == 0;
        int64_t dilation = (Data[offset++] % 2) + 1;      // 1-2 dilation

        // Calculate valid groups (must divide both in_channels and out_channels)
        if (groups_selector > 0) {
            for (int64_t g = std::min(in_channels, out_channels); g >= 1; g--) {
                if (in_channels % g == 0 && out_channels % g == 0) {
                    if (groups_selector == 1 || g == 1) {
                        groups = g;
                        break;
                    }
                    groups_selector--;
                }
            }
        }

        // Parse spatial dimensions
        int64_t batch_size = (Data[offset++] % 3) + 1;    // 1-3 batch
        int64_t depth = (Data[offset++] % 4) + kernel_size;  // ensure >= kernel
        int64_t height = (Data[offset++] % 4) + kernel_size;
        int64_t width = (Data[offset++] % 4) + kernel_size;

        // Create input tensor with correct shape: [N, C_in, D, H, W]
        torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width});

        // Create ConvTranspose3d module
        torch::nn::ConvTranspose3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding_val)
               .groups(groups)
               .bias(use_bias)
               .dilation(dilation);

        auto conv_transpose = torch::nn::ConvTranspose3d(options);

        // Apply the forward pass
        torch::Tensor output;
        try {
            output = conv_transpose->forward(input);
        } catch (...) {
            // Shape mismatch or other expected errors - silently continue
        }

        // Test with output_size parameter
        if (offset < Size && output.defined()) {
            try {
                // ConvTranspose3d can take an output_size to resolve ambiguity
                std::vector<int64_t> output_size = {
                    output.size(2) + (Data[offset] % 2),
                    output.size(3) + ((offset + 1 < Size) ? (Data[offset + 1] % 2) : 0),
                    output.size(4) + ((offset + 2 < Size) ? (Data[offset + 2] % 2) : 0)
                };
                offset += 3;
                output = conv_transpose->forward(input, output_size);
            } catch (...) {
                // Invalid output_size - silently continue
            }
        }

        // Test with double precision
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                auto input_double = input.to(torch::kDouble);
                auto conv_transpose_double = torch::nn::ConvTranspose3d(options);
                conv_transpose_double->to(torch::kDouble);
                output = conv_transpose_double->forward(input_double);
            } catch (...) {
                // Type conversion issues - silently continue
            }
        }

        // Test with different batch sizes
        if (offset < Size) {
            try {
                int64_t new_batch_size = (Data[offset++] % 4) + 1;
                auto new_input = torch::randn({new_batch_size, in_channels, depth, height, width});
                output = conv_transpose->forward(new_input);
            } catch (...) {
                // Silently continue
            }
        }

        // Test with varied spatial dimensions
        if (offset + 2 < Size) {
            try {
                int64_t new_depth = (Data[offset++] % 6) + kernel_size;
                int64_t new_height = (Data[offset++] % 6) + kernel_size;
                int64_t new_width = (Data[offset++] % 6) + kernel_size;
                auto varied_input = torch::randn({batch_size, in_channels, new_depth, new_height, new_width});
                output = conv_transpose->forward(varied_input);
            } catch (...) {
                // Silently continue
            }
        }

        // Test gradient computation
        if (offset < Size && Data[offset++] % 3 == 0 && output.defined()) {
            try {
                auto grad_input = input.clone().requires_grad_(true);
                auto grad_output = conv_transpose->forward(grad_input);
                auto loss = grad_output.sum();
                loss.backward();
            } catch (...) {
                // Gradient issues - silently continue
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}