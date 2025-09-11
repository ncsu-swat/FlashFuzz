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
        
        // Early exit if not enough data
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
                channels = (Data[offset++] % 8) + 1;
                depth = (Data[offset++] % 8) + 1;
                height = (Data[offset++] % 8) + 1;
                width = (Data[offset++] % 8) + 1;
            }
            
            input = input.reshape({batch_size, channels, depth, height, width});
        }
        
        // Extract parameters for Conv3d from the input data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Use remaining bytes to set parameters if available
        if (offset + 7 <= Size) {
            out_channels = (Data[offset++] % 8) + 1;
            kernel_size = (Data[offset++] % 5) + 1;
            stride = (Data[offset++] % 3) + 1;
            padding = Data[offset++] % 3;
            dilation = (Data[offset++] % 2) + 1;
            groups = Data[offset++] % std::max(1, static_cast<int>(in_channels));
            if (groups > 1) {
                // Ensure out_channels is divisible by groups
                out_channels = groups * ((out_channels / groups) + 1);
            }
            bias = Data[offset++] % 2 == 0;
        }
        
        // Create regular Conv3d module (QAT modules are not available in C++ frontend)
        torch::nn::Conv3dOptions conv_options = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
            
        torch::nn::Conv3d conv3d(conv_options);
        
        // Set to training mode
        conv3d->train();
        
        // Forward pass
        torch::Tensor output = conv3d->forward(input);
        
        // Try different input types
        if (offset < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            if (dtype != input.scalar_type()) {
                try {
                    torch::Tensor input2 = input.to(dtype);
                    torch::Tensor output2 = conv3d->forward(input2);
                } catch (const std::exception& e) {
                    // Ignore exceptions from type conversion
                }
            }
        }
        
        // Try with different batch sizes
        if (offset < Size && input.size(0) > 1) {
            try {
                torch::Tensor single_batch = input.slice(0, 0, 1);
                torch::Tensor output_single = conv3d->forward(single_batch);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
        }
        
        // Try with different parameters
        if (offset + 1 < Size) {
            try {
                // Create another conv3d with different parameters
                int64_t new_out_channels = (Data[offset++] % 8) + 1;
                torch::nn::Conv3dOptions new_conv_options = torch::nn::Conv3dOptions(in_channels, new_out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(1)
                    .bias(bias);
                
                torch::nn::Conv3d new_conv3d(new_conv_options);
                new_conv3d->train();
                torch::Tensor output3 = new_conv3d->forward(input);
            } catch (const std::exception& e) {
                // Ignore exceptions
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
