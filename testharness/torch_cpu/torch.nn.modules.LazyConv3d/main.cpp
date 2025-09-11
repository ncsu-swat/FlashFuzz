#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (batch, channels, depth, height, width)
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 5) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 5) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for Conv3d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        if (offset + 8 <= Size) {
            out_channels = static_cast<int64_t>(Data[offset++]) % 16 + 1;
            
            kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            
            stride = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            
            padding = static_cast<int64_t>(Data[offset++]) % 3;
            
            dilation = static_cast<int64_t>(Data[offset++]) % 2 + 1;
            
            groups = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            
            bias = Data[offset++] % 2 == 0;
        }
        
        // Ensure groups divides in_channels
        if (in_channels % groups != 0) {
            groups = 1;
        }
        
        // Create Conv3d module
        torch::nn::Conv3d conv(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                               .stride(stride)
                               .padding(padding)
                               .dilation(dilation)
                               .groups(groups)
                               .bias(bias));
        
        // Apply the module to the input tensor
        torch::Tensor output = conv->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Access the weight and bias if available
        if (bias) {
            auto bias_tensor = conv->bias;
            if (bias_tensor.defined()) {
                sum += bias_tensor.sum();
            }
        }
        
        auto weight_tensor = conv->weight;
        if (weight_tensor.defined()) {
            sum += weight_tensor.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
