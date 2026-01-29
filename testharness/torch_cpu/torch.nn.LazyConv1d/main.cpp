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
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for Conv1d (N, C, L)
        if (input.dim() < 3) {
            int64_t numel = input.numel();
            if (numel < 1) {
                return 0;
            }
            input = input.reshape({1, 1, numel});
        }
        
        // Ensure input length is sufficient for convolution
        if (input.size(2) < 1) {
            return 0;
        }
        
        // Get in_channels from input tensor
        int64_t in_channels = input.size(1);
        
        // Extract parameters for Conv1d
        int64_t out_channels = 1 + (offset < Size ? Data[offset++] % 16 : 1);
        int64_t kernel_size = 1 + (offset < Size ? Data[offset++] % 7 : 1);
        int64_t stride = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        int64_t padding = offset < Size ? Data[offset++] % 3 : 0;
        int64_t dilation = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t groups = 1;
        
        // Try to set groups to a divisor of in_channels and out_channels if possible
        if (offset < Size && in_channels > 1) {
            int64_t proposed_groups = 1 + (Data[offset++] % in_channels);
            if (in_channels % proposed_groups == 0 && out_channels % proposed_groups == 0) {
                groups = proposed_groups;
            }
        }
        
        bool bias = offset < Size ? (Data[offset++] % 2 == 0) : true;
        
        // Ensure kernel size doesn't exceed input length considering dilation
        int64_t effective_kernel = dilation * (kernel_size - 1) + 1;
        int64_t input_length = input.size(2);
        if (effective_kernel > input_length + 2 * padding) {
            // Adjust kernel size to be valid
            kernel_size = 1;
        }
        
        // Create Conv1d module with explicit in_channels
        // Note: LazyConv1d is Python-only, so we use Conv1d in C++
        auto conv_options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                .stride(stride)
                                .padding(padding)
                                .dilation(dilation)
                                .groups(groups)
                                .bias(bias);
        
        torch::nn::Conv1d conv(conv_options);
        
        // Apply the Conv1d operation
        torch::Tensor output = conv->forward(input);
        
        // Force computation to materialize the tensor
        output = output.clone();
        
        // Access some values to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
        
        // Test multiple forward passes with different inputs
        if (offset < Size && Data[offset] % 4 == 0) {
            offset++;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input2.dim() < 3) {
                int64_t numel2 = input2.numel();
                if (numel2 >= 1) {
                    input2 = input2.reshape({1, in_channels, numel2});
                    // Inner try-catch for expected shape mismatches
                    try {
                        torch::Tensor output2 = conv->forward(input2);
                        (void)output2;
                    } catch (...) {
                        // Shape mismatch expected, ignore silently
                    }
                }
            }
        }
        
        // Test with different padding modes if available
        if (offset < Size && Data[offset] % 3 == 0) {
            offset++;
            auto conv_reflect = torch::nn::Conv1d(
                torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
                    .bias(bias)
                    .padding_mode(torch::kReflect));
            
            try {
                torch::Tensor output_reflect = conv_reflect->forward(input);
                (void)output_reflect;
            } catch (...) {
                // Padding mode may not work with all configurations
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}