#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for ConvTranspose1d (N, C, L)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvTranspose1d
        int64_t in_channels = input.size(1);
        
        // Parse out_channels
        int64_t out_channels = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1; // Ensure positive and reasonable
        }
        
        // Parse kernel_size
        int64_t kernel_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 7 + 1; // Ensure positive and reasonable
        }
        
        // Parse stride
        int64_t stride = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 5 + 1; // Ensure positive and reasonable
        }
        
        // Parse padding
        int64_t padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 4; // Ensure non-negative and reasonable
        }
        
        // Parse output_padding
        int64_t output_padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_padding = std::abs(output_padding) % 3; // Ensure non-negative and reasonable
        }
        
        // Parse groups
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % 4 + 1; // Ensure positive and reasonable
        }
        
        // Parse bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Parse dilation
        int64_t dilation = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 3 + 1; // Ensure positive and reasonable
        }
        
        // Create ConvTranspose1d module
        torch::nn::ConvTranspose1dOptions options = 
            torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .groups(groups)
                .bias(bias)
                .dilation(dilation);
        
        auto module = torch::nn::ConvTranspose1d(options);
        
        // Forward pass
        torch::Tensor output = module->forward(input);
        
        // Access some properties of the output to ensure computation
        auto output_size = output.sizes();
        auto output_sum = output.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}