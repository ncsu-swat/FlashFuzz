#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (N, C, D, H, W) for Conv3d
        if (input.dim() < 5) {
            // Expand dimensions if needed
            while (input.dim() < 5) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for Conv3d from the remaining data
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        std::vector<int64_t> kernel_size;
        std::vector<int64_t> stride;
        std::vector<int64_t> padding;
        std::vector<int64_t> dilation;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse in_channels and out_channels
        if (offset + 2 < Size) {
            in_channels = std::max(int64_t(1), int64_t(Data[offset++]));
            out_channels = std::max(int64_t(1), int64_t(Data[offset++]));
        } else {
            in_channels = 1;
            out_channels = 1;
        }
        
        // Parse kernel_size (3D)
        for (int i = 0; i < 3; i++) {
            if (offset < Size) {
                kernel_size.push_back(std::max(int64_t(1), int64_t(Data[offset++])));
            } else {
                kernel_size.push_back(1);
            }
        }
        
        // Parse stride (3D)
        for (int i = 0; i < 3; i++) {
            if (offset < Size) {
                stride.push_back(std::max(int64_t(1), int64_t(Data[offset++])));
            } else {
                stride.push_back(1);
            }
        }
        
        // Parse padding (3D)
        for (int i = 0; i < 3; i++) {
            if (offset < Size) {
                padding.push_back(int64_t(Data[offset++]));
            } else {
                padding.push_back(0);
            }
        }
        
        // Parse dilation (3D)
        for (int i = 0; i < 3; i++) {
            if (offset < Size) {
                dilation.push_back(std::max(int64_t(1), int64_t(Data[offset++])));
            } else {
                dilation.push_back(1);
            }
        }
        
        // Parse groups
        if (offset < Size) {
            groups = std::max(int64_t(1), int64_t(Data[offset++]) % in_channels);
            if (groups > 1) {
                // Ensure in_channels is divisible by groups
                in_channels = groups * (in_channels / groups);
                if (in_channels == 0) in_channels = groups;
            }
        }
        
        // Parse bias
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        // Ensure input tensor has the correct number of channels
        if (input.size(1) != in_channels) {
            // Resize the input tensor to have the correct number of channels
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[1] = in_channels;
            input = torch::zeros(new_shape, input.options());
        }
        
        // Create Conv3d module (using regular Conv3d instead of LazyConv3d)
        torch::nn::Conv3d conv(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Apply the Conv3d operation
        torch::Tensor output = conv(input);
        
        // Force materialization of the output
        output = output.clone();
        
        // Access some elements to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            if (std::isnan(sum) || std::isinf(sum)) {
                return 0;
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