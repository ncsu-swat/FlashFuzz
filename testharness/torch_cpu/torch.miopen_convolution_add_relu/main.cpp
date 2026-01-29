#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Minimum size needed for meaningful input
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Parse convolution parameters first from raw bytes
        // to ensure deterministic tensor shapes

        // Get spatial dimensions (1D, 2D, or 3D convolution)
        uint8_t spatial_dims = (Data[offset++] % 2) + 1; // 1 or 2 (2D or 3D conv)
        
        // Parse batch size and channels
        uint8_t batch_size = (Data[offset++] % 4) + 1;      // 1-4
        uint8_t in_channels = (Data[offset++] % 4) + 1;     // 1-4
        uint8_t out_channels = (Data[offset++] % 4) + 1;    // 1-4
        
        // Parse spatial sizes
        std::vector<int64_t> spatial_size;
        for (int i = 0; i < spatial_dims && offset < Size; i++) {
            spatial_size.push_back((Data[offset++] % 8) + 4); // 4-11
        }
        
        // Parse kernel size
        std::vector<int64_t> kernel_size;
        for (int i = 0; i < spatial_dims && offset < Size; i++) {
            kernel_size.push_back((Data[offset++] % 3) + 1); // 1-3
        }
        
        // Ensure we have valid sizes
        while (spatial_size.size() < static_cast<size_t>(spatial_dims)) {
            spatial_size.push_back(8);
        }
        while (kernel_size.size() < static_cast<size_t>(spatial_dims)) {
            kernel_size.push_back(3);
        }

        // Parse stride
        std::vector<int64_t> stride;
        for (int i = 0; i < spatial_dims && offset < Size; i++) {
            stride.push_back((Data[offset++] % 2) + 1); // 1-2
        }
        while (stride.size() < static_cast<size_t>(spatial_dims)) {
            stride.push_back(1);
        }

        // Parse padding
        std::vector<int64_t> padding;
        for (int i = 0; i < spatial_dims && offset < Size; i++) {
            padding.push_back(Data[offset++] % 2); // 0-1
        }
        while (padding.size() < static_cast<size_t>(spatial_dims)) {
            padding.push_back(0);
        }

        // Parse dilation
        std::vector<int64_t> dilation;
        for (int i = 0; i < spatial_dims && offset < Size; i++) {
            dilation.push_back((Data[offset++] % 2) + 1); // 1-2
        }
        while (dilation.size() < static_cast<size_t>(spatial_dims)) {
            dilation.push_back(1);
        }

        // Parse groups (must divide both in_channels and out_channels)
        int64_t groups = 1;
        if (offset < Size) {
            uint8_t g = Data[offset++] % 4 + 1;
            // Find valid group count
            for (int64_t candidate = g; candidate >= 1; candidate--) {
                if (in_channels % candidate == 0 && out_channels % candidate == 0) {
                    groups = candidate;
                    break;
                }
            }
        }

        // Parse alpha for the addition
        float alpha_val = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp alpha to reasonable range
            if (std::isnan(alpha_val) || std::isinf(alpha_val)) {
                alpha_val = 1.0f;
            }
            alpha_val = std::max(-10.0f, std::min(10.0f, alpha_val));
        }

        // Build input shape: [N, C_in, spatial...]
        std::vector<int64_t> input_shape = {batch_size, in_channels};
        input_shape.insert(input_shape.end(), spatial_size.begin(), spatial_size.end());

        // Build weight shape: [C_out, C_in/groups, kernel...]
        std::vector<int64_t> weight_shape = {out_channels, in_channels / groups};
        weight_shape.insert(weight_shape.end(), kernel_size.begin(), kernel_size.end());

        // Create tensors with proper shapes
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input = torch::randn(input_shape, options);
        torch::Tensor weight = torch::randn(weight_shape, options);
        torch::Tensor bias = torch::randn({out_channels}, options);

        // Calculate output spatial dimensions for z tensor
        std::vector<int64_t> output_spatial;
        for (int i = 0; i < spatial_dims; i++) {
            int64_t out_dim = (spatial_size[i] + 2 * padding[i] - 
                              dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1;
            if (out_dim <= 0) {
                return 0; // Invalid configuration
            }
            output_spatial.push_back(out_dim);
        }

        // z tensor must match output shape for the add operation
        std::vector<int64_t> z_shape = {batch_size, out_channels};
        z_shape.insert(z_shape.end(), output_spatial.begin(), output_spatial.end());
        torch::Tensor z = torch::randn(z_shape, options);

        // miopen_convolution_add_relu is AMD ROCm/MIOpen specific
        // Check if we have HIP (ROCm) available
        #ifdef USE_ROCM
        if (torch::cuda::is_available()) {
            // Move tensors to GPU
            input = input.cuda();
            weight = weight.cuda();
            bias = bias.cuda();
            z = z.cuda();

            try {
                at::Scalar alpha(alpha_val);
                torch::Tensor output = torch::miopen_convolution_add_relu(
                    input, weight, z, alpha, bias, stride, padding, dilation, groups);
                
                // Basic sanity check on output
                if (output.numel() > 0) {
                    (void)output.sum().item<float>();
                }
            } catch (const c10::Error& e) {
                // Expected exceptions from shape mismatches, etc.
            }
        }
        #else
        // On non-ROCm builds, we can still test the CPU path fails gracefully
        // or test with regular convolution + add + relu as a fallback
        try {
            // This will likely throw since miopen is not available
            at::Scalar alpha(alpha_val);
            torch::Tensor output = torch::miopen_convolution_add_relu(
                input, weight, z, alpha, bias, stride, padding, dilation, groups);
        } catch (const c10::Error& e) {
            // Expected - MIOpen not available on this platform
        }
        #endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}