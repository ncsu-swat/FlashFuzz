#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    // This API requires CUDA with cuDNN support
    if (!torch::cuda::is_available() || !torch::cuda::cudnn_is_available()) {
        // Cannot test this API without CUDA+cuDNN
        return 0;
    }

    try
    {
        size_t offset = 0;
        
        // Need enough data for parameters
        if (Size < 16) {
            return 0;
        }
        
        // Parse convolution parameters from fuzzer data
        int64_t batch_size = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        int64_t in_channels = static_cast<int64_t>(Data[offset++]) % 16 + 1;
        int64_t out_channels = static_cast<int64_t>(Data[offset++]) % 16 + 1;
        int64_t height = static_cast<int64_t>(Data[offset++]) % 32 + 4;
        int64_t width = static_cast<int64_t>(Data[offset++]) % 32 + 4;
        int64_t kernel_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
        int64_t kernel_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
        int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 3;
        int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 3;
        int64_t dilation_h = static_cast<int64_t>(Data[offset++]) % 2 + 1;
        int64_t dilation_w = static_cast<int64_t>(Data[offset++]) % 2 + 1;
        
        // Groups must divide both in_channels and out_channels
        int64_t groups = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        // Adjust groups to be a common divisor
        while (groups > 1 && (in_channels % groups != 0 || out_channels % groups != 0)) {
            groups--;
        }
        
        bool use_bias = (Data[offset++] % 2) == 1;
        uint8_t dtype_selector = Data[offset++] % 2;
        
        // Select dtype (cuDNN conv supports float and half)
        auto dtype = (dtype_selector == 0) ? torch::kFloat32 : torch::kFloat16;
        auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
        
        // Create input tensor: (N, C_in, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width}, options);
        
        // Create weight tensor: (C_out, C_in/groups, kH, kW)
        torch::Tensor weight = torch::randn({out_channels, in_channels / groups, kernel_h, kernel_w}, options);
        
        // Optional bias tensor: (C_out,)
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (use_bias) {
            bias = torch::randn({out_channels}, options);
        }
        
        std::vector<int64_t> stride = {stride_h, stride_w};
        std::vector<int64_t> padding = {padding_h, padding_w};
        std::vector<int64_t> dilation = {dilation_h, dilation_w};
        
        // Validate that output size would be positive
        int64_t out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int64_t out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        
        if (out_h <= 0 || out_w <= 0) {
            // Invalid configuration, skip
            return 0;
        }
        
        // Call the API under test
        torch::Tensor output = torch::cudnn_convolution_relu(
            input, weight, bias,
            stride, padding, dilation, groups
        );
        
        // Use the output to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch/c10 errors (shape mismatches, cuDNN errors, etc.)
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}