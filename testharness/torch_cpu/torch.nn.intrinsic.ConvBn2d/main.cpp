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
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for Conv2d (N, C, H, W)
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 1, 1});
        }
        
        // Extract parameters for Conv2d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool bias = true;
        
        // If we have more data, use it to set parameters
        if (offset + 7 < Size) {
            out_channels = (Data[offset] % 8) + 1;
            kernel_size = (Data[offset + 1] % 5) + 1;
            stride = (Data[offset + 2] % 3) + 1;
            padding = Data[offset + 3] % 3;
            dilation = (Data[offset + 4] % 2) + 1;
            bias = Data[offset + 5] % 2 == 0;
            offset += 6;
        }
        
        // Create Conv2d module
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(
                in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .bias(bias)
        );
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(out_channels);
        
        // Initialize weights with random values
        for (auto& p : conv->parameters()) {
            p.data().uniform_(-0.1, 0.1);
        }
        for (auto& p : bn->parameters()) {
            p.data().uniform_(-0.1, 0.1);
        }
        
        // Set modules to evaluation mode
        conv->eval();
        bn->eval();
        
        // Apply the modules to the input tensor (conv followed by bn)
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Try to access output properties to ensure computation completed
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try to perform some operations on the output
        if (offset < Size) {
            uint8_t op_selector = Data[offset++];
            
            switch (op_selector % 4) {
                case 0:
                    output = output.sigmoid();
                    break;
                case 1:
                    output = output.relu();
                    break;
                case 2:
                    output = output.tanh();
                    break;
                case 3:
                    output = output.abs();
                    break;
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
