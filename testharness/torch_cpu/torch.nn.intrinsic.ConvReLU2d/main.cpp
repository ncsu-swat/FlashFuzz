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
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        
        // Ensure we have at least 4 dimensions (N, C, H, W)
        while (input.dim() < 4) {
            input = input.unsqueeze(-1);
        }
        
        // Get input dimensions
        int64_t batch_size = input.size(0);
        int64_t in_channels = input.size(1);
        int64_t in_height = input.size(2);
        int64_t in_width = input.size(3);
        
        // Ensure we have at least 1 channel
        if (in_channels < 1) {
            input = torch::ones({batch_size, 1, in_height, in_width}, input.options());
            in_channels = 1;
        }
        
        // Parse parameters for ConvReLU2d
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Extract parameters from remaining data if available
        if (offset + 7 <= Size) {
            out_channels = (Data[offset] % 8) + 1;  // 1-8 output channels
            kernel_size = (Data[offset + 1] % 5) + 1;  // 1-5 kernel size
            stride = (Data[offset + 2] % 3) + 1;  // 1-3 stride
            padding = Data[offset + 3] % 3;  // 0-2 padding
            dilation = (Data[offset + 4] % 2) + 1;  // 1-2 dilation
            groups = (Data[offset + 5] % in_channels) + 1;  // 1-in_channels groups
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                groups = 1;
            }
            
            // Ensure out_channels is divisible by groups
            if (out_channels % groups != 0) {
                out_channels = groups;
            }
            
            bias = (Data[offset + 6] % 2) == 1;  // true or false
            offset += 7;
        }
        
        // Create Conv2d module and manually apply ReLU (since intrinsic ConvReLU2d is not available)
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
            
        auto conv = torch::nn::Conv2d(conv_options);
        
        // Convert input to float if needed
        if (input.dtype() != torch::kFloat && 
            input.dtype() != torch::kDouble && 
            input.dtype() != torch::kHalf) {
            input = input.to(torch::kFloat);
        }
        
        // Apply Conv2d followed by ReLU to simulate ConvReLU2d
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = torch::relu(conv_output);
        
        // Verify output has ReLU applied (no negative values)
        auto min_val = torch::min(output).item<float>();
        if (min_val < 0) {
            throw std::runtime_error("ConvReLU2d output contains negative values");
        }
        
        // Test with eval mode
        conv->eval();
        torch::Tensor conv_output_eval = conv->forward(input);
        torch::Tensor output_eval = torch::relu(conv_output_eval);
        
        // Test with train mode
        conv->train();
        torch::Tensor conv_output_train = conv->forward(input);
        torch::Tensor output_train = torch::relu(conv_output_train);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
