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
        
        // Ensure input has at least 5 dimensions (N, C, D, H, W)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, 
                                  std::max<int64_t>(1, input.numel() > 0 ? input.size(0) : 1),
                                  std::max<int64_t>(1, input.numel() > 1 ? input.size(-1) : 1),
                                  std::max<int64_t>(1, input.numel() > 2 ? input.size(-2) : 1)});
        } else if (input.dim() < 5) {
            std::vector<int64_t> new_shape;
            // Add batch dimension if needed
            if (input.dim() == 3) {
                new_shape = {1, 1, input.size(0), input.size(1), input.size(2)};
            } else if (input.dim() == 4) {
                new_shape = {1, input.size(0), input.size(1), input.size(2), input.size(3)};
            }
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for ConvTranspose3d from the input data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t groups = 1;
        bool bias = true;
        int64_t dilation = 1;
        
        // Parse additional parameters if data is available
        if (offset + 8 <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            kernel_size = (Data[offset + 1] % 5) + 1;
            stride = (Data[offset + 2] % 3) + 1;
            padding = Data[offset + 3] % 3;
            output_padding = Data[offset + 4] % 2;
            
            // Ensure groups divides both in_channels and out_channels
            groups = (Data[offset + 5] % std::min(in_channels, out_channels)) + 1;
            if (groups > 1) {
                // Adjust in_channels and out_channels to be divisible by groups
                in_channels = in_channels - (in_channels % groups);
                if (in_channels == 0) in_channels = groups;
                out_channels = out_channels - (out_channels % groups);
                if (out_channels == 0) out_channels = groups;
                
                // Reshape input tensor to match adjusted in_channels
                auto old_shape = input.sizes().vec();
                old_shape[1] = in_channels;
                input = input.reshape(old_shape);
            }
            
            bias = (Data[offset + 6] % 2) == 0;
            dilation = (Data[offset + 7] % 3) + 1;
            offset += 8;
        }
        
        // Create ConvTranspose3d module
        torch::nn::ConvTranspose3d conv_transpose(
            torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .groups(groups)
                .bias(bias)
                .dilation(dilation)
        );
        
        // Apply the module to the input tensor
        torch::Tensor output = conv_transpose->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Try different input shapes if there's more data
        if (offset + 10 < Size) {
            // Create another input tensor with different shape
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Reshape to match expected input dimensions
            if (input2.dim() < 3) {
                input2 = input2.reshape({1, in_channels, 
                                       std::max<int64_t>(1, input2.numel() > 0 ? input2.size(0) : 1),
                                       std::max<int64_t>(1, input2.numel() > 1 ? input2.size(-1) : 1),
                                       std::max<int64_t>(1, input2.numel() > 2 ? input2.size(-2) : 1)});
            } else if (input2.dim() < 5) {
                std::vector<int64_t> new_shape;
                if (input2.dim() == 3) {
                    new_shape = {1, in_channels, input2.size(0), input2.size(1), input2.size(2)};
                } else if (input2.dim() == 4) {
                    new_shape = {1, in_channels, input2.size(1), input2.size(2), input2.size(3)};
                }
                input2 = input2.reshape(new_shape);
            } else {
                // Ensure channel dimension matches in_channels
                auto shape = input2.sizes().vec();
                shape[1] = in_channels;
                input2 = input2.reshape(shape);
            }
            
            // Apply the module to the second input
            torch::Tensor output2 = conv_transpose->forward(input2);
            sum += output2.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}