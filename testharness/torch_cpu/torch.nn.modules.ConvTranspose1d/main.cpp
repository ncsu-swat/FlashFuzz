#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is float type for convolution
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has at least 3 dimensions (batch_size, channels, length)
        if (input.dim() < 3) {
            int64_t numel = input.numel();
            if (numel == 0) {
                numel = 1;
            }
            input = input.reshape({1, 1, numel});
        }
        
        // Get in_channels from input tensor
        int64_t in_channels = input.size(1);
        if (in_channels <= 0) {
            in_channels = 1;
        }
        
        // Parse parameters from data with safe bounds
        auto readByte = [&]() -> uint8_t {
            if (offset < Size) {
                return Data[offset++];
            }
            return 0;
        };
        
        // Extract parameters for ConvTranspose1d
        int64_t out_channels = (readByte() % 16) + 1;  // 1-16
        int64_t kernel_size = (readByte() % 7) + 1;    // 1-7
        int64_t stride = (readByte() % 4) + 1;         // 1-4
        int64_t padding = readByte() % 5;              // 0-4
        int64_t dilation = (readByte() % 3) + 1;       // 1-3
        bool bias = readByte() & 1;
        
        // output_padding must be less than max(stride, dilation)
        int64_t max_output_padding = std::max(stride, dilation);
        int64_t output_padding = readByte() % max_output_padding;
        
        // Handle groups - must divide both in_channels and out_channels
        int64_t groups = 1;
        uint8_t groups_byte = readByte();
        // Try to find a valid groups value
        for (int64_t g = (groups_byte % in_channels) + 1; g >= 1; g--) {
            if (in_channels % g == 0 && out_channels % g == 0) {
                groups = g;
                break;
            }
        }
        
        // Create ConvTranspose1d module
        torch::nn::ConvTranspose1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride);
        options.padding(padding);
        options.output_padding(output_padding);
        options.dilation(dilation);
        options.groups(groups);
        options.bias(bias);
        
        torch::nn::ConvTranspose1d conv_transpose(options);
        
        // Apply the convolution
        torch::Tensor output = conv_transpose->forward(input);
        
        // Try different input types
        if (offset < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            try {
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || dtype == torch::kFloat16) {
                    torch::Tensor input_cast = input.to(dtype);
                    torch::Tensor output_cast = conv_transpose->forward(input_cast);
                }
            } catch (const std::exception &) {
                // Ignore exceptions from type conversion - expected for some dtypes
            }
        }
        
        // Try with different batch sizes
        if (input.size(0) > 1) {
            try {
                torch::Tensor slice_batch = input.slice(0, 0, 1);
                torch::Tensor output_slice = conv_transpose->forward(slice_batch);
            } catch (const std::exception &) {
                // Ignore exceptions
            }
        }
        
        // Try eval mode
        try {
            conv_transpose->eval();
            torch::Tensor output_eval = conv_transpose->forward(input);
        } catch (const std::exception &) {
            // Ignore exceptions
        }
        
        // Test with no_grad
        {
            torch::NoGradGuard no_grad;
            try {
                torch::Tensor output_no_grad = conv_transpose->forward(input);
            } catch (const std::exception &) {
                // Ignore exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}