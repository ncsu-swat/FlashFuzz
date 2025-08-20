#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (batch, channels, depth, height, width)
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvTranspose3d from the remaining data
        int64_t in_channels = 1;
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t groups = 1;
        bool bias = true;
        int64_t dilation = 1;
        
        // Parse parameters if we have enough data
        if (offset + 9 <= Size) {
            in_channels = (Data[offset++] % 4) + 1;  // 1-4 channels
            out_channels = (Data[offset++] % 4) + 1; // 1-4 channels
            kernel_size = (Data[offset++] % 3) + 1;  // 1-3 kernel size
            stride = (Data[offset++] % 3) + 1;       // 1-3 stride
            padding = Data[offset++] % 3;            // 0-2 padding
            output_padding = Data[offset++] % 2;     // 0-1 output padding
            groups = (Data[offset++] % in_channels) + 1; // 1 to in_channels
            if (groups > 1) {
                // Ensure in_channels is divisible by groups
                in_channels = groups * ((in_channels / groups) + 1);
                // Ensure out_channels is divisible by groups
                out_channels = groups * ((out_channels / groups) + 1);
            }
            bias = Data[offset++] % 2 == 0;          // 50% chance of bias
            dilation = (Data[offset++] % 2) + 1;     // 1-2 dilation
        }
        
        // Ensure input has correct number of channels
        if (input.size(1) != in_channels) {
            input = input.expand({input.size(0), in_channels, input.size(2), input.size(3), input.size(4)});
        }
        
        // Create ConvTranspose3d module
        torch::nn::ConvTranspose3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .groups(groups)
               .bias(bias)
               .dilation(dilation);
        
        auto conv_transpose = torch::nn::ConvTranspose3d(options);
        
        // Apply the operation
        torch::Tensor output = conv_transpose->forward(input);
        
        // Optionally test with different input types
        if (offset < Size) {
            auto dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try with a different data type if supported for this operation
            if (dtype == torch::kFloat || dtype == torch::kDouble) {
                input = input.to(dtype);
                conv_transpose = torch::nn::ConvTranspose3d(options);
                output = conv_transpose->forward(input);
            }
        }
        
        // Test with different batch sizes
        if (offset < Size && input.size(0) > 0) {
            int64_t new_batch_size = (Data[offset++] % 3) + 1; // 1-3 batch size
            if (new_batch_size != input.size(0)) {
                auto resized_input = input.repeat({new_batch_size, 1, 1, 1, 1});
                output = conv_transpose->forward(resized_input);
            }
        }
        
        // Test with different spatial dimensions
        if (offset + 2 < Size) {
            int64_t depth = (Data[offset++] % 5) + 1;    // 1-5 depth
            int64_t height = (Data[offset++] % 5) + 1;   // 1-5 height
            int64_t width = (Data[offset++] % 5) + 1;    // 1-5 width
            
            if (depth != input.size(2) || height != input.size(3) || width != input.size(4)) {
                auto resized_input = torch::zeros({input.size(0), in_channels, depth, height, width}, 
                                                 input.options());
                output = conv_transpose->forward(resized_input);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}