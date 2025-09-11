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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for Conv3d
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t batch_size = 1;
            int64_t channels = 3;
            int64_t depth = 4;
            int64_t height = 4;
            int64_t width = 4;
            
            // Use some bytes from input to determine dimensions if available
            if (offset + 5 <= Size) {
                batch_size = (Data[offset++] % 3) + 1;
                channels = (Data[offset++] % 4) + 1;
                depth = (Data[offset++] % 5) + 1;
                height = (Data[offset++] % 5) + 1;
                width = (Data[offset++] % 5) + 1;
            }
            
            input = input.reshape({batch_size, channels, depth, height, width});
        }
        
        // Extract parameters for Conv3d + BatchNorm3d combination
        int64_t in_channels = input.size(1);
        int64_t out_channels = 2;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 1;
        int64_t dilation = 1;
        bool bias = true;
        
        // Use remaining bytes to set parameters if available
        if (offset + 6 <= Size) {
            out_channels = (Data[offset++] % 8) + 1;
            kernel_size = (Data[offset++] % 3) + 1;
            stride = (Data[offset++] % 2) + 1;
            padding = Data[offset++] % 2;
            dilation = (Data[offset++] % 2) + 1;
            bias = Data[offset++] % 2 == 0;
        }
        
        // Create Conv3d and BatchNorm3d modules separately since intrinsic module is not available
        torch::nn::Conv3d conv(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                                   .stride(stride)
                                   .padding(padding)
                                   .dilation(dilation)
                                   .bias(bias));
        
        torch::nn::BatchNorm3d bn(torch::nn::BatchNorm3dOptions(out_channels));
        
        // Set modules to evaluation mode
        conv->eval();
        bn->eval();
        
        // Apply the modules to the input tensor (conv followed by bn)
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Test with different input types
        if (offset < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            if (dtype != input.scalar_type()) {
                try {
                    torch::Tensor input_cast = input.to(dtype);
                    torch::Tensor conv_output_cast = conv->forward(input_cast);
                    torch::Tensor output_cast = bn->forward(conv_output_cast);
                } catch (const std::exception&) {
                    // Some dtype conversions might not be valid, that's fine
                }
            }
        }
        
        // Test with different batch sizes
        if (offset < Size && input.size(0) > 1) {
            try {
                torch::Tensor single_batch = input.slice(0, 0, 1);
                torch::Tensor conv_single = conv->forward(single_batch);
                torch::Tensor output_single = bn->forward(conv_single);
            } catch (const std::exception&) {
                // Handle potential errors
            }
        }
        
        // Test with zero-sized dimensions if possible
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor zero_batch = torch::zeros({0, in_channels, input.size(2), input.size(3), input.size(4)}, 
                                                       input.options());
                torch::Tensor conv_zero = conv->forward(zero_batch);
                torch::Tensor output_zero = bn->forward(conv_zero);
            } catch (const std::exception&) {
                // This might throw, which is expected
            }
        }
        
        // Test training mode
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                conv->train();
                bn->train();
                torch::Tensor conv_train = conv->forward(input);
                torch::Tensor output_train = bn->forward(conv_train);
            } catch (const std::exception&) {
                // Handle potential errors
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
