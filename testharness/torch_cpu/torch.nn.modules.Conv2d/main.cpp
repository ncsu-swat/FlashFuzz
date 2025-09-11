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
        
        // Extract parameters for Conv2d from the remaining data
        uint8_t in_channels = 0;
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t dilation = 0;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset < Size) in_channels = Data[offset++] % 16 + 1;
        if (offset < Size) out_channels = Data[offset++] % 16 + 1;
        if (offset < Size) kernel_size = Data[offset++] % 7 + 1;
        if (offset < Size) stride = Data[offset++] % 4 + 1;
        if (offset < Size) padding = Data[offset++] % 4;
        if (offset < Size) dilation = Data[offset++] % 3 + 1;
        if (offset < Size) groups = Data[offset++] % std::max(1, static_cast<int>(in_channels)) + 1;
        if (offset < Size) bias = Data[offset++] % 2 == 0;
        
        // Ensure groups divides in_channels
        if (in_channels % groups != 0) {
            groups = 1;
        }
        
        // Create Conv2d module
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Reshape input tensor to match Conv2d requirements if needed
        auto input_shape = input.sizes().vec();
        if (input_shape.size() >= 3) {
            // Ensure batch dimension exists
            if (input_shape.size() == 3) {
                input = input.unsqueeze(0);
                input_shape.insert(input_shape.begin(), 1);
            }
            
            // Set channel dimension to match in_channels
            input_shape[1] = in_channels;
            
            // Ensure spatial dimensions are large enough for the convolution
            int min_spatial_size = (kernel_size - 1) * dilation + 1 - 2 * padding;
            if (min_spatial_size < 1) min_spatial_size = 1;
            
            for (size_t i = 2; i < input_shape.size(); i++) {
                if (input_shape[i] < min_spatial_size) {
                    input_shape[i] = min_spatial_size;
                }
            }
            
            // Resize the input tensor
            input = input.resize_(input_shape);
        }
        
        // Forward pass
        torch::Tensor output = conv->forward(input);
        
        // Access some properties to ensure computation happened
        auto output_size = output.sizes();
        auto output_sum = output.sum().item<float>();
        
        // Test backward pass
        if (output.requires_grad()) {
            auto grad = torch::ones_like(output);
            output.backward(grad);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
