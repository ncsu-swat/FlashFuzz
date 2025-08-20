#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
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
        
        // Ensure input has at least 3 dimensions (N, C, L) for Conv1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        uint8_t stride = 1, padding = 0, dilation = 1, groups = 1;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 input channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 output channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            groups = Data[offset++];               // Groups
            
            // Ensure groups is valid (must be a divisor of in_channels and out_channels)
            if (groups == 0) groups = 1;
            else groups = std::min(groups, std::min(in_channels, out_channels));
            
            // Ensure in_channels and out_channels are divisible by groups
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
            
            out_channels = (out_channels / groups) * groups;
            if (out_channels == 0) out_channels = groups;
        }
        
        // Reshape input to match in_channels if needed
        if (input.size(1) != in_channels) {
            auto shape = input.sizes().vec();
            shape[1] = in_channels;
            input = input.reshape(shape);
        }
        
        // Create Conv1d module
        torch::nn::Conv1d conv(
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(true)
        );
        
        // Create weight and bias tensors
        auto weight = torch::randn({out_channels, in_channels / groups, kernel_size});
        auto bias = torch::randn({out_channels});
        
        // Set weight and bias
        conv->weight.data() = weight;
        conv->bias.data() = bias;
        
        // Forward pass
        auto output = conv->forward(input);
        
        // Apply ReLU activation
        output = torch::relu(output);
        
        // Try to access some properties to ensure they're valid
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}