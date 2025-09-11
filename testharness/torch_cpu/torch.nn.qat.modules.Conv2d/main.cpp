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
        
        // Ensure input has at least 4 dimensions for Conv2d (N, C, H, W)
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 
                                  input.numel() > 0 ? input.numel() : 1, 
                                  1});
        }
        
        // Extract parameters for Conv2d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        uint8_t kernel_size = 0, stride = 0, padding = 0, dilation = 0;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset < Size) in_channels = Data[offset++] % 8 + 1;
        if (offset < Size) out_channels = Data[offset++] % 8 + 1;
        if (offset < Size) kernel_size = Data[offset++] % 5 + 1;
        if (offset < Size) stride = Data[offset++] % 3 + 1;
        if (offset < Size) padding = Data[offset++] % 3;
        if (offset < Size) dilation = Data[offset++] % 2 + 1;
        if (offset < Size) groups = Data[offset++] % std::max(static_cast<int>(1), std::min(static_cast<int>(in_channels), static_cast<int>(out_channels))) + 1;
        if (offset < Size) bias = Data[offset++] % 2 == 0;
        
        // Ensure in_channels is divisible by groups
        if (in_channels % groups != 0) {
            in_channels = groups;
        }
        
        // Ensure out_channels is divisible by groups
        if (out_channels % groups != 0) {
            out_channels = groups;
        }
        
        // Adjust input tensor to match in_channels
        if (input.size(1) != in_channels) {
            auto old_shape = input.sizes().vec();
            old_shape[1] = in_channels;
            input = input.reshape(old_shape);
        }
        
        // Create regular Conv2d module (QAT modules are not available in C++ frontend)
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
            
        torch::nn::Conv2d conv(conv_options);
        
        // Set to training mode
        conv->train();
        
        // Set scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        if (offset + 8 < Size) {
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        if (offset + 8 < Size) {
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scale is positive and zero_point is in valid range
        scale = std::abs(scale);
        if (scale < 1e-10) scale = 1e-10;
        zero_point = std::max(std::min(zero_point, static_cast<int64_t>(127)), static_cast<int64_t>(-128));
        
        // Forward pass
        torch::Tensor output;
        try {
            output = conv->forward(input);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
        
        // Try backward pass if possible
        if (output.requires_grad()) {
            try {
                auto sum = output.sum();
                sum.backward();
            } catch (const c10::Error& e) {
                // Backward might fail
                return 0;
            }
        }
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}
