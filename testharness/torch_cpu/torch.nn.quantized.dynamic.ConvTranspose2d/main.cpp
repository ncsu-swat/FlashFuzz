#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W)
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for ConvTranspose2d
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse parameters from input data
        if (offset + 8 <= Size) {
            memcpy(&in_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + 8 <= Size) {
            memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + 8 <= Size) {
            memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + 8 <= Size) {
            memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + 8 <= Size) {
            memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + 8 <= Size) {
            memcpy(&output_padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + 8 <= Size) {
            memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + 8 <= Size) {
            memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        // Ensure parameters are within reasonable bounds
        in_channels = std::abs(in_channels) % 64 + 1;
        out_channels = std::abs(out_channels) % 64 + 1;
        kernel_size = std::abs(kernel_size) % 7 + 1;
        stride = std::abs(stride) % 4 + 1;
        padding = std::abs(padding) % 4;
        output_padding = std::abs(output_padding) % stride;
        dilation = std::abs(dilation) % 3 + 1;
        groups = std::abs(groups) % std::min(in_channels, out_channels) + 1;
        
        // Ensure in_channels is divisible by groups
        in_channels = (in_channels / groups) * groups;
        if (in_channels == 0) in_channels = groups;
        
        // Ensure out_channels is divisible by groups
        out_channels = (out_channels / groups) * groups;
        if (out_channels == 0) out_channels = groups;
        
        // Create ConvTranspose2d module using regular (non-quantized) version
        torch::nn::ConvTranspose2dOptions options(
            in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose2d(options);
        
        // Reshape input tensor to match expected input shape for ConvTranspose2d
        if (input.dim() > 4) {
            input = input.flatten(0, input.dim() - 4);
        }
        
        // Ensure input has correct number of channels
        if (input.size(1) != in_channels) {
            auto shape = input.sizes().vec();
            shape[1] = in_channels;
            input = input.reshape(shape);
        }
        
        // Apply the operation
        auto output = conv_transpose(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
