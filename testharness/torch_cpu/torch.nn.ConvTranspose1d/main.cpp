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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (batch_size, channels, length)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvTranspose1d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse parameters if we have more data
        if (offset + 8 <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            offset++;
            
            kernel_size = (Data[offset] % 5) + 1;
            offset++;
            
            stride = (Data[offset] % 3) + 1;
            offset++;
            
            padding = Data[offset] % 3;
            offset++;
            
            output_padding = Data[offset] % 2;
            offset++;
            
            dilation = (Data[offset] % 2) + 1;
            offset++;
            
            // Ensure groups divides in_channels
            if (in_channels > 0) {
                groups = (Data[offset] % in_channels) + 1;
            }
            offset++;
            
            bias = Data[offset] % 2 == 0;
            offset++;
        }
        
        // Ensure parameters are valid
        if (in_channels > 0 && groups > 0) {
            // Ensure in_channels is divisible by groups
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
            
            // Ensure out_channels is divisible by groups
            out_channels = (out_channels / groups) * groups;
            if (out_channels == 0) out_channels = groups;
        }
        
        // Create ConvTranspose1d module
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
        
        // Ensure the output is valid
        if (output.numel() > 0) {
            // Access some elements to ensure computation happened
            float sum = output.sum().item<float>();
            (void)sum;  // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
