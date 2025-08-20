#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early return if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (batch_size, channels, depth, height, width)
        if (input.dim() != 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvTranspose3d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        
        // Get kernel size from data if available
        int64_t kernel_size = 3;
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 5 + 1; // Keep kernel size reasonable (1-5)
        }
        
        // Get stride from data if available
        int64_t stride = 1;
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1; // Keep stride reasonable (1-3)
        }
        
        // Get padding from data if available
        int64_t padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Keep padding reasonable (0-2)
        }
        
        // Get output_padding from data if available
        int64_t output_padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&output_padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_padding = std::abs(output_padding) % 2; // Keep output_padding reasonable (0-1)
        }
        
        // Get dilation from data if available
        int64_t dilation = 1;
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 2 + 1; // Keep dilation reasonable (1-2)
        }
        
        // Get groups from data if available
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1; // Keep groups reasonable (1 to in_channels)
            if (groups > in_channels) groups = in_channels;
            if (in_channels % groups != 0) groups = 1; // Ensure in_channels is divisible by groups
        }
        
        // Get bias flag from data if available
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create the ConvTranspose3d module (using regular conv_transpose3d)
        torch::nn::ConvTranspose3dOptions options(
            in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose3d(options);
        
        // Apply the operation
        auto output = conv_transpose->forward(input);
        
        // Optionally test other methods
        if (offset < Size && (Data[offset++] & 0x1)) {
            auto weight = conv_transpose->weight;
            if (conv_transpose->options.bias()) {
                auto bias_tensor = conv_transpose->bias;
            }
        }
        
        // Test with different input types
        if (offset < Size && (Data[offset++] & 0x1)) {
            // Create a different input tensor
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure input has 5 dimensions
            if (input2.dim() != 5) {
                input2 = input2.reshape({1, 1, 1, 1, 1});
                
                // Ensure channel dimension matches in_channels
                if (input2.size(1) != in_channels) {
                    input2 = input2.expand({input2.size(0), in_channels, input2.size(2), input2.size(3), input2.size(4)});
                }
            }
            
            // Apply the operation again
            auto output2 = conv_transpose->forward(input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}