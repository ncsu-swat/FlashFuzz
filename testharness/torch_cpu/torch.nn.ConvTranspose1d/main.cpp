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

    // Disable gradient computation for fuzzing
    torch::NoGradGuard no_grad;

    try
    {
        // Need at least some data to proceed
        if (Size < 12) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters first so we can create proper input tensor
        int64_t in_channels = (Data[offset] % 8) + 1;
        offset++;

        int64_t out_channels = (Data[offset] % 8) + 1;
        offset++;

        int64_t kernel_size = (Data[offset] % 5) + 1;
        offset++;

        int64_t stride = (Data[offset] % 3) + 1;
        offset++;

        int64_t padding = Data[offset] % 3;
        offset++;

        int64_t dilation = (Data[offset] % 2) + 1;
        offset++;

        // output_padding must be strictly less than max(stride, dilation)
        int64_t max_output_padding = std::max(stride, dilation) - 1;
        int64_t output_padding = 0;
        if (max_output_padding > 0) {
            output_padding = Data[offset] % (max_output_padding + 1);
        }
        offset++;

        // groups must divide both in_channels and out_channels
        int64_t groups = (Data[offset] % 4) + 1;
        offset++;

        // Adjust channels to be divisible by groups
        in_channels = ((in_channels + groups - 1) / groups) * groups;
        out_channels = ((out_channels + groups - 1) / groups) * groups;

        bool bias = Data[offset] % 2 == 0;
        offset++;

        // Get batch size and sequence length
        int64_t batch_size = (Data[offset] % 4) + 1;
        offset++;

        int64_t seq_length = (Data[offset] % 16) + 1;
        offset++;

        // Create input tensor with correct shape: (batch_size, in_channels, length)
        torch::Tensor input = torch::randn({batch_size, in_channels, seq_length});

        // If we have more data, use it to influence tensor values
        if (offset < Size) {
            torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            // Use fuzz data to scale input
            float scale = (fuzz_tensor.numel() > 0) ? 
                          (fuzz_tensor.abs().mean().item<float>() + 0.1f) : 1.0f;
            input = input * scale;
        }

        // Create ConvTranspose1d module with the parameters
        auto options = torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);

        auto conv_transpose = torch::nn::ConvTranspose1d(options);

        // Apply the operation
        torch::Tensor output = conv_transpose->forward(input);

        // Verify output is valid
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum;
        }

        // Test with output_size parameter (alternate forward signature)
        try {
            std::vector<int64_t> output_size = {output.size(2)};
            torch::Tensor output2 = conv_transpose->forward(input, output_size);
            if (output2.numel() > 0) {
                float sum2 = output2.sum().item<float>();
                (void)sum2;
            }
        }
        catch (...) {
            // Silently ignore - output_size variant may fail with certain params
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}