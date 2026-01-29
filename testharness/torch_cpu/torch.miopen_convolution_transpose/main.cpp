#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // torch::miopen_convolution_transpose requires AMD GPU with MIOpen/ROCm
        // This API is not available on CPU - skip if no MIOpen support
        
        // Check if MIOpen is available (requires CUDA/HIP device)
        if (!torch::cuda::is_available()) {
            // MIOpen requires AMD GPU, silently skip on CPU-only systems
            return 0;
        }
        
        size_t offset = 0;
        
        // Need sufficient data for tensor creation and parameters
        if (Size < 20) {
            return 0;
        }
        
        // Parse convolution dimensions (1D, 2D, or 3D)
        int spatial_dims = (Data[offset++] % 3) + 1; // 1, 2, or 3
        
        // Create batch size and channels from fuzz data
        int64_t batch_size = (Data[offset++] % 4) + 1;      // 1-4
        int64_t in_channels = (Data[offset++] % 8) + 1;     // 1-8
        int64_t out_channels = (Data[offset++] % 8) + 1;    // 1-8
        
        // Parse groups - must divide both in_channels and out_channels
        int64_t groups = (Data[offset++] % 4) + 1;
        // Adjust channels to be divisible by groups
        in_channels = ((in_channels + groups - 1) / groups) * groups;
        out_channels = ((out_channels + groups - 1) / groups) * groups;
        
        // Build input shape: [N, C_in, *spatial_dims]
        std::vector<int64_t> input_shape = {batch_size, in_channels};
        for (int i = 0; i < spatial_dims && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 8) + 4; // 4-11
            input_shape.push_back(dim_size);
        }
        while (input_shape.size() < static_cast<size_t>(2 + spatial_dims)) {
            input_shape.push_back(4);
        }
        
        // Build kernel size
        std::vector<int64_t> kernel_size;
        for (int i = 0; i < spatial_dims && offset < Size; i++) {
            kernel_size.push_back((Data[offset++] % 3) + 1); // 1-3
        }
        while (kernel_size.size() < static_cast<size_t>(spatial_dims)) {
            kernel_size.push_back(3);
        }
        
        // Build weight shape: [C_in, C_out/groups, *kernel_size]
        std::vector<int64_t> weight_shape = {in_channels, out_channels / groups};
        weight_shape.insert(weight_shape.end(), kernel_size.begin(), kernel_size.end());
        
        // Parse stride, padding, output_padding, dilation
        std::vector<int64_t> stride, padding, output_padding, dilation;
        for (int i = 0; i < spatial_dims; i++) {
            stride.push_back(offset < Size ? (Data[offset++] % 2) + 1 : 1);
            padding.push_back(offset < Size ? Data[offset++] % 2 : 0);
            output_padding.push_back(offset < Size ? Data[offset++] % 2 : 0);
            dilation.push_back(offset < Size ? (Data[offset++] % 2) + 1 : 1);
        }
        
        // Ensure output_padding < stride (required by convolution transpose)
        for (size_t i = 0; i < output_padding.size(); i++) {
            if (output_padding[i] >= stride[i]) {
                output_padding[i] = stride[i] - 1;
            }
        }
        
        // Parse flags
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        bool benchmark = offset < Size && Data[offset++] % 2 == 0;
        bool deterministic = offset < Size && Data[offset++] % 2 == 0;
        
        // Create tensors on GPU (required for MIOpen)
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA);
        
        torch::Tensor input = torch::randn(input_shape, options);
        torch::Tensor weight = torch::randn(weight_shape, options);
        torch::Tensor bias;
        if (use_bias) {
            bias = torch::randn({out_channels}, options);
        }
        
        // Call miopen_convolution_transpose
        torch::Tensor output;
        try {
            output = torch::miopen_convolution_transpose(
                input, weight, bias,
                padding, output_padding, stride, dilation,
                groups, benchmark, deterministic
            );
        } catch (const c10::Error& e) {
            // Expected failures due to invalid parameter combinations
            return 0;
        }
        
        // Use the output to prevent optimization
        auto sum = output.sum();
        volatile float result = sum.item<float>();
        (void)result;
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}