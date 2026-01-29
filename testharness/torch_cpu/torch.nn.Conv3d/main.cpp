#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 20) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point (Conv3d requires float)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has 5 dimensions (batch, channels, depth, height, width)
        while (input.dim() < 5) {
            input = input.unsqueeze(0);
        }
        
        // Ensure we have at least 1 in each dimension and reasonable sizes
        std::vector<int64_t> input_sizes;
        for (int i = 0; i < 5; i++) {
            input_sizes.push_back(std::max(input.size(i), int64_t(1)));
        }
        
        // Reshape to ensure minimum sizes
        if (input.size(0) < 1 || input.size(1) < 1 || 
            input.size(2) < 1 || input.size(3) < 1 || input.size(4) < 1) {
            input = torch::ones({1, 1, 4, 4, 4}, torch::kFloat32);
        }
        
        int64_t in_channels = input.size(1);
        int64_t depth = input.size(2);
        int64_t height = input.size(3);
        int64_t width = input.size(4);
        
        // Extract parameters for Conv3d from the remaining data
        int64_t out_channels = 1;
        std::vector<int64_t> kernel_size = {1, 1, 1};
        std::vector<int64_t> stride = {1, 1, 1};
        std::vector<int64_t> padding = {0, 0, 0};
        std::vector<int64_t> dilation = {1, 1, 1};
        int64_t groups = 1;
        bool bias = true;
        
        // Parse remaining parameters if data available
        if (offset + 1 <= Size) {
            out_channels = std::max(int64_t(1), int64_t(Data[offset++] % 16) + 1);
        }
        
        // Parse kernel size - ensure it fits in the input
        if (offset + 3 <= Size) {
            kernel_size[0] = std::max(int64_t(1), std::min(int64_t(Data[offset++] % 4) + 1, depth));
            kernel_size[1] = std::max(int64_t(1), std::min(int64_t(Data[offset++] % 4) + 1, height));
            kernel_size[2] = std::max(int64_t(1), std::min(int64_t(Data[offset++] % 4) + 1, width));
        }
        
        // Parse stride
        if (offset + 3 <= Size) {
            stride[0] = std::max(int64_t(1), int64_t(Data[offset++] % 3) + 1);
            stride[1] = std::max(int64_t(1), int64_t(Data[offset++] % 3) + 1);
            stride[2] = std::max(int64_t(1), int64_t(Data[offset++] % 3) + 1);
        }
        
        // Parse padding - limit to reasonable values
        if (offset + 3 <= Size) {
            padding[0] = int64_t(Data[offset++] % 3);
            padding[1] = int64_t(Data[offset++] % 3);
            padding[2] = int64_t(Data[offset++] % 3);
        }
        
        // Parse dilation
        if (offset + 3 <= Size) {
            dilation[0] = std::max(int64_t(1), int64_t(Data[offset++] % 2) + 1);
            dilation[1] = std::max(int64_t(1), int64_t(Data[offset++] % 2) + 1);
            dilation[2] = std::max(int64_t(1), int64_t(Data[offset++] % 2) + 1);
        }
        
        // Parse groups - must divide in_channels evenly
        if (offset < Size) {
            int64_t max_groups = in_channels;
            groups = std::max(int64_t(1), int64_t(Data[offset++]) % (max_groups + 1));
            
            // Ensure in_channels is divisible by groups
            while (groups > 1 && in_channels % groups != 0) {
                groups--;
            }
        }
        
        // Parse bias
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        // Validate that output size would be positive
        // output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        auto calc_output_size = [](int64_t input_size, int64_t pad, int64_t dil, int64_t kernel, int64_t str) {
            return (input_size + 2 * pad - dil * (kernel - 1) - 1) / str + 1;
        };
        
        int64_t out_d = calc_output_size(depth, padding[0], dilation[0], kernel_size[0], stride[0]);
        int64_t out_h = calc_output_size(height, padding[1], dilation[1], kernel_size[1], stride[1]);
        int64_t out_w = calc_output_size(width, padding[2], dilation[2], kernel_size[2], stride[2]);
        
        if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
            // Invalid combination, skip silently
            return 0;
        }
        
        // Create Conv3d module
        torch::nn::Conv3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv = torch::nn::Conv3d(options);
        
        // Test forward pass
        torch::Tensor output = conv->forward(input);
        
        // Test backward pass for better coverage
        if (output.numel() > 0 && output.requires_grad()) {
            try {
                output.sum().backward();
            } catch (...) {
                // Backward might fail for some configurations, that's OK
            }
        }
        
        // Also test with different padding modes
        if (offset < Size && Data[offset] % 3 == 0) {
            torch::nn::Conv3dOptions options2(in_channels, out_channels, kernel_size);
            options2.stride(stride)
                   .padding(padding)
                   .dilation(dilation)
                   .groups(groups)
                   .bias(bias)
                   .padding_mode(torch::kZeros);
            
            auto conv2 = torch::nn::Conv3d(options2);
            torch::Tensor output2 = conv2->forward(input);
            volatile float sum = output2.sum().item<float>();
            (void)sum;
        }
        
        // Ensure we don't optimize away the computation
        volatile float sum = output.sum().item<float>();
        (void)sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}