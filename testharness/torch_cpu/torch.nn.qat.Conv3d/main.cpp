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
        
        // Early return if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (N, C, D, H, W) for Conv3d
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for Conv3d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        
        if (offset + 1 < Size) {
            out_channels = (Data[offset++] % 8) + 1; // 1-8 output channels
        }
        
        // Parse kernel size
        std::vector<int64_t> kernel_size(3, 1);
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            kernel_size[i] = (Data[offset++] % 3) + 1; // 1-3 kernel size
        }
        
        // Parse stride
        std::vector<int64_t> stride(3, 1);
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            stride[i] = (Data[offset++] % 3) + 1; // 1-3 stride
        }
        
        // Parse padding
        std::vector<int64_t> padding(3, 0);
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            padding[i] = Data[offset++] % 3; // 0-2 padding
        }
        
        // Parse dilation
        std::vector<int64_t> dilation(3, 1);
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            dilation[i] = (Data[offset++] % 2) + 1; // 1-2 dilation
        }
        
        // Parse groups
        int64_t groups = 1;
        if (offset < Size) {
            groups = (Data[offset++] % in_channels) + 1;
            if (groups > in_channels) groups = in_channels;
            if (in_channels % groups != 0) groups = 1;
        }
        
        // Parse bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        // Parse padding_mode
        torch::nn::detail::conv_padding_mode_t padding_mode = torch::kZeros;
        if (offset < Size) {
            uint8_t mode_selector = Data[offset++] % 3;
            switch (mode_selector) {
                case 0: padding_mode = torch::kZeros; break;
                case 1: padding_mode = torch::kReflect; break;
                case 2: padding_mode = torch::kReplicate; break;
            }
        }
        
        // Create Conv3d module (using regular Conv3d as QAT is not available in C++ frontend)
        torch::nn::Conv3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias)
               .padding_mode(padding_mode);
        
        torch::nn::Conv3d conv3d(options);
        
        // Set scale and zero_point for quantization (simulate QAT behavior)
        double scale = 1.0;
        int64_t zero_point = 0;
        if (offset + 8 <= Size) {
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (scale <= 0.0) scale = 1.0;
        }
        
        if (offset + 8 <= Size) {
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply the Conv3d operation
        torch::Tensor output = conv3d(input);
        
        // Ensure we use the output to prevent optimization
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
