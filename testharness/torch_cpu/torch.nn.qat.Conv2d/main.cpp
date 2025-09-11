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
        
        // Early exit for very small inputs
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for Conv2d from the remaining data
        int64_t in_channels = 1;
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        if (offset + 8 <= Size) {
            in_channels = (Data[offset] % 8) + 1;
            offset++;
            out_channels = (Data[offset] % 8) + 1;
            offset++;
            kernel_size = (Data[offset] % 5) + 1;
            offset++;
            stride = (Data[offset] % 3) + 1;
            offset++;
            padding = Data[offset] % 3;
            offset++;
            dilation = (Data[offset] % 2) + 1;
            offset++;
            groups = std::gcd(in_channels, out_channels) > 1 ? std::gcd(in_channels, out_channels) : 1;
            offset++;
            bias = Data[offset] % 2 == 0;
            offset++;
        }
        
        // Create regular Conv2d module (QAT modules are not available in C++ frontend)
        torch::nn::Conv2dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        torch::nn::Conv2d conv2d(options);
        
        // Reshape input tensor to match expected input shape for Conv2d
        // Ensure channel dimension matches in_channels
        std::vector<int64_t> input_shape = input.sizes().vec();
        if (input_shape.size() >= 2) {
            input_shape[1] = in_channels;
            
            // Ensure spatial dimensions are at least kernel_size
            for (size_t i = 2; i < input_shape.size(); i++) {
                if (input_shape[i] < kernel_size) {
                    input_shape[i] = kernel_size;
                }
            }
            
            // Resize input tensor
            input = input.resize_(input_shape);
        }
        
        // Forward pass
        torch::Tensor output = conv2d->forward(input);
        
        // Try to access some properties to ensure computation happened
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
