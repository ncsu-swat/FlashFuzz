#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Ensure we have enough data left for parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Extract parameters for the conv module
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Read parameters from input data
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&in_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_channels = std::abs(in_channels) % 64 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 64 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 7 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 4 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Ensure non-negative and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 3 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1; // Ensure positive and reasonable
            
            // Ensure in_channels is divisible by groups
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
        }
        
        if (offset < Size) {
            bias = Data[offset++] & 1; // Use 1 bit for bias
        }
        
        // Determine the type of convolution based on input dimensions
        if (input.dim() >= 3) {
            // Try different types of convolution based on input dimensions
            if (input.dim() == 3) {
                // 1D Convolution
                torch::nn::Conv1d conv(
                    torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .groups(groups)
                        .bias(bias)
                );
                
                // Ensure input has correct shape for Conv1d
                if (input.dim() == 3 && input.size(1) == in_channels) {
                    auto output = conv(input);
                }
            }
            else if (input.dim() == 4) {
                // 2D Convolution
                torch::nn::Conv2d conv(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .groups(groups)
                        .bias(bias)
                );
                
                // Ensure input has correct shape for Conv2d
                if (input.dim() == 4 && input.size(1) == in_channels) {
                    auto output = conv(input);
                }
            }
            else if (input.dim() == 5) {
                // 3D Convolution
                torch::nn::Conv3d conv(
                    torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .groups(groups)
                        .bias(bias)
                );
                
                // Ensure input has correct shape for Conv3d
                if (input.dim() == 5 && input.size(1) == in_channels) {
                    auto output = conv(input);
                }
            }
        }
        
        // Try with different input shapes if the original doesn't work
        if (input.dim() < 3 || input.size(1) != in_channels) {
            // Create a new tensor with appropriate shape for Conv2d
            std::vector<int64_t> shape = {1, in_channels, 28, 28}; // Example shape
            torch::Tensor new_input = torch::ones(shape, input.options());
            
            // 2D Convolution
            torch::nn::Conv2d conv(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
                    .bias(bias)
            );
            
            auto output = conv(new_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
