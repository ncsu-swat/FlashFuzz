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
        
        // Parse convolution parameters from the remaining data
        uint8_t in_channels = 0;
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t dilation = 0;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 16 + 1;
            out_channels = Data[offset++] % 16 + 1;
            kernel_size = Data[offset++] % 7 + 1;
            stride = Data[offset++] % 5 + 1;
            padding = Data[offset++] % 4;
            dilation = Data[offset++] % 3 + 1;
            groups = Data[offset++] % in_channels + 1;
            
            if (groups > in_channels) {
                groups = in_channels;
            }
            
            if (in_channels % groups != 0) {
                in_channels = groups * (in_channels / groups + 1);
            }
            
            if (offset < Size) {
                bias = Data[offset++] % 2 == 0;
            }
        }
        
        // Reshape input tensor if needed to match convolution requirements
        auto input_sizes = input.sizes().vec();
        int64_t batch_size = 1;
        
        if (input_sizes.empty()) {
            // Scalar tensor, reshape to 4D
            input = input.reshape({1, in_channels, kernel_size, kernel_size});
        } else if (input_sizes.size() == 1) {
            // 1D tensor, reshape to 4D
            batch_size = input_sizes[0] > 0 ? input_sizes[0] : 1;
            input = input.reshape({batch_size, in_channels, kernel_size, kernel_size});
        } else if (input_sizes.size() == 2) {
            // 2D tensor, reshape to 4D
            batch_size = input_sizes[0] > 0 ? input_sizes[0] : 1;
            int64_t features = input_sizes[1] > 0 ? input_sizes[1] : in_channels;
            input = input.reshape({batch_size, features, kernel_size, kernel_size});
        } else if (input_sizes.size() == 3) {
            // 3D tensor, reshape to 4D for Conv2d
            batch_size = input_sizes[0] > 0 ? input_sizes[0] : 1;
            int64_t features = input_sizes[1] > 0 ? input_sizes[1] : in_channels;
            int64_t height = input_sizes[2] > 0 ? input_sizes[2] : kernel_size;
            input = input.reshape({batch_size, features, height, kernel_size});
        } else if (input_sizes.size() > 4) {
            // Higher dimensional tensor, reshape to 4D
            batch_size = input_sizes[0] > 0 ? input_sizes[0] : 1;
            int64_t features = input_sizes[1] > 0 ? input_sizes[1] : in_channels;
            int64_t height = input_sizes[2] > 0 ? input_sizes[2] : kernel_size;
            int64_t width = input_sizes[3] > 0 ? input_sizes[3] : kernel_size;
            input = input.reshape({batch_size, features, height, width});
        }
        
        // Ensure input has the right number of channels for the convolution
        input_sizes = input.sizes().vec();
        if (input_sizes.size() >= 2 && input_sizes[1] != in_channels) {
            input_sizes[1] = in_channels;
            input = input.reshape(input_sizes);
        }
        
        // Create Conv1d, Conv2d, or Conv3d based on input dimensions
        if (input.dim() == 3) {
            // Conv1d
            torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                    .stride(stride)
                                    .padding(padding)
                                    .dilation(dilation)
                                    .groups(groups)
                                    .bias(bias));
            
            auto output = conv->forward(input);
        } else if (input.dim() == 4) {
            // Conv2d
            torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                    .stride(stride)
                                    .padding(padding)
                                    .dilation(dilation)
                                    .groups(groups)
                                    .bias(bias));
            
            auto output = conv->forward(input);
        } else if (input.dim() == 5) {
            // Conv3d
            torch::nn::Conv3d conv(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                                    .stride(stride)
                                    .padding(padding)
                                    .dilation(dilation)
                                    .groups(groups)
                                    .bias(bias));
            
            auto output = conv->forward(input);
        }
        
        // Try with different kernel sizes for height and width
        if (offset + 1 < Size && input.dim() == 4) {
            uint8_t kernel_h = Data[offset++] % 5 + 1;
            uint8_t kernel_w = Data[offset++] % 5 + 1;
            
            torch::nn::Conv2d conv2(torch::nn::Conv2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
                                    .stride({stride, stride})
                                    .padding({padding, padding})
                                    .dilation({dilation, dilation})
                                    .groups(groups)
                                    .bias(bias));
            
            auto output = conv2->forward(input);
        }
        
        // Try transposed convolution
        if (offset < Size && input.dim() == 4) {
            torch::nn::ConvTranspose2d conv_t(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .output_padding(padding > 0 ? padding - 1 : 0)
                    .groups(groups)
                    .bias(bias)
                    .dilation(dilation));
            
            auto output = conv_t->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}