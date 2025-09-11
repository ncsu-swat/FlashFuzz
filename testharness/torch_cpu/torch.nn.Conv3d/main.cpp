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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (batch, channels, depth, height, width)
        if (input.dim() < 5) {
            // Expand dimensions if needed
            while (input.dim() < 5) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for Conv3d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        std::vector<int64_t> kernel_size = {1, 1, 1};
        std::vector<int64_t> stride = {1, 1, 1};
        std::vector<int64_t> padding = {0, 0, 0};
        std::vector<int64_t> dilation = {1, 1, 1};
        int64_t groups = 1;
        bool bias = true;
        
        // Parse remaining parameters if data available
        if (offset + 8 <= Size) {
            out_channels = std::max(int64_t(1), int64_t(Data[offset++]));
            
            // Parse kernel size
            if (offset + 3 <= Size) {
                for (int i = 0; i < 3; i++) {
                    kernel_size[i] = std::max(int64_t(1), int64_t(Data[offset++]));
                }
            }
            
            // Parse stride
            if (offset + 3 <= Size) {
                for (int i = 0; i < 3; i++) {
                    stride[i] = std::max(int64_t(1), int64_t(Data[offset++]));
                }
            }
            
            // Parse padding
            if (offset + 3 <= Size) {
                for (int i = 0; i < 3; i++) {
                    padding[i] = int64_t(Data[offset++]);
                }
            }
            
            // Parse dilation
            if (offset + 3 <= Size) {
                for (int i = 0; i < 3; i++) {
                    dilation[i] = std::max(int64_t(1), int64_t(Data[offset++]));
                }
            }
            
            // Parse groups
            if (offset < Size) {
                groups = std::max(int64_t(1), int64_t(Data[offset++]) % (in_channels + 1));
                
                // Ensure in_channels is divisible by groups
                if (in_channels % groups != 0) {
                    groups = 1;
                }
            }
            
            // Parse bias
            if (offset < Size) {
                bias = Data[offset++] % 2 == 0;
            }
        }
        
        // Create Conv3d module
        torch::nn::Conv3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv = torch::nn::Conv3d(options);
        
        // Apply the Conv3d operation
        torch::Tensor output = conv->forward(input);
        
        // Ensure we don't optimize away the computation
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
