#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some data to proceed
        if (Size < 20) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters from the input data first
        int64_t in_channels = (Data[offset++] % 16) + 1; // 1-16 channels
        int64_t out_channels = (Data[offset++] % 16) + 1; // 1-16 channels
        int64_t kernel_size = (Data[offset++] % 5) + 1; // 1-5 kernel size
        int64_t stride = (Data[offset++] % 3) + 1; // 1-3 stride
        int64_t padding = Data[offset++] % (kernel_size); // 0-(kernel_size-1) padding
        int64_t output_padding = Data[offset++] % stride; // 0-(stride-1) output padding
        int64_t groups = (Data[offset++] % 4) + 1; // 1-4 groups
        bool use_bias = Data[offset++] % 2 == 0; // 50% chance of bias
        int64_t dilation = (Data[offset++] % 2) + 1; // 1-2 dilation
        
        // Ensure in_channels and out_channels are divisible by groups
        in_channels = ((in_channels + groups - 1) / groups) * groups;
        out_channels = ((out_channels + groups - 1) / groups) * groups;
        
        // Ensure in_channels and out_channels are at least groups
        if (in_channels < groups) in_channels = groups;
        if (out_channels < groups) out_channels = groups;
        
        // Get spatial dimensions from fuzz data
        int64_t batch_size = (Data[offset++] % 4) + 1; // 1-4
        int64_t height = (Data[offset++] % 8) + 4; // 4-11
        int64_t width = (Data[offset++] % 8) + 4; // 4-11
        
        // Create ConvTranspose2d module
        auto options = torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .groups(groups)
            .bias(use_bias)
            .dilation(dilation);
        
        torch::nn::ConvTranspose2d conv_transpose(options);
        
        // Create input tensor with proper shape for ConvTranspose2d
        // ConvTranspose2d expects input of shape [batch_size, in_channels, height, width]
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});
        
        // Use remaining fuzz data to perturb the input values
        if (offset < Size) {
            size_t remaining = Size - offset;
            auto input_accessor = input.flatten();
            int64_t num_elements = input_accessor.numel();
            for (size_t i = 0; i < remaining && i < static_cast<size_t>(num_elements); i++) {
                // Scale the byte to a small perturbation
                float perturbation = (static_cast<float>(Data[offset + i]) - 128.0f) / 128.0f;
                input_accessor[i] = input_accessor[i] + perturbation;
            }
            input = input_accessor.reshape({batch_size, in_channels, height, width});
        }
        
        // Apply the ConvTranspose2d operation
        torch::Tensor output = conv_transpose->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Test backward pass
        try {
            input.set_requires_grad(true);
            torch::Tensor input_grad = torch::randn({batch_size, in_channels, height, width});
            input_grad.set_requires_grad(true);
            
            torch::nn::ConvTranspose2d conv_transpose_grad(options);
            torch::Tensor output_grad = conv_transpose_grad->forward(input_grad);
            output_grad.sum().backward();
        } catch (...) {
            // Silently ignore gradient computation failures
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}