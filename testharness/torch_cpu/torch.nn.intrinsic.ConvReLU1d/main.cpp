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
        
        // Early return if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for ConvReLU1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvReLU1d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // If we have more data, use it to set parameters
        if (offset + 8 <= Size) {
            // Parse out_channels (1-64)
            out_channels = (Data[offset] % 64) + 1;
            offset++;
            
            // Parse kernel_size (1-7)
            kernel_size = (Data[offset] % 7) + 1;
            offset++;
            
            // Parse stride (1-3)
            stride = (Data[offset] % 3) + 1;
            offset++;
            
            // Parse padding (0-3)
            padding = Data[offset] % 4;
            offset++;
            
            // Parse dilation (1-2)
            dilation = (Data[offset] % 2) + 1;
            offset++;
            
            // Parse groups (1 to in_channels)
            if (in_channels > 0) {
                groups = (Data[offset] % in_channels) + 1;
                // Ensure in_channels is divisible by groups
                in_channels = (in_channels / groups) * groups;
                if (in_channels == 0) in_channels = groups;
            } else {
                groups = 1;
                in_channels = 1;
            }
            offset++;
            
            // Parse bias
            bias = (Data[offset] % 2) == 1;
            offset++;
        }
        
        // Create Conv1d and ReLU modules separately
        torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                   .stride(stride)
                                   .padding(padding)
                                   .dilation(dilation)
                                   .groups(groups)
                                   .bias(bias));
        
        // Forward pass through Conv1d
        torch::Tensor conv_output = conv->forward(input);
        
        // Apply ReLU
        torch::Tensor output = torch::relu(conv_output);
        
        // Verify output has expected properties
        if (output.dim() != 3) {
            throw std::runtime_error("Output dimension mismatch");
        }
        
        // Verify ReLU effect (no negative values)
        torch::Tensor neg_values = output < 0;
        if (neg_values.any().item<bool>()) {
            throw std::runtime_error("Output contains negative values after ReLU");
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
