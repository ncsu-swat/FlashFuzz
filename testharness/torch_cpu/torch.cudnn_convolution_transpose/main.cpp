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

    try
    {
        // cudnn_convolution_transpose requires CUDA
        if (!torch::cuda::is_available()) {
            return 0;
        }

        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Parse convolution parameters first to determine tensor shapes
        int64_t batch_size = (static_cast<int64_t>(Data[offset++]) % 4) + 1;
        int64_t in_channels = (static_cast<int64_t>(Data[offset++]) % 8) + 1;
        int64_t out_channels = (static_cast<int64_t>(Data[offset++]) % 8) + 1;
        int64_t height = (static_cast<int64_t>(Data[offset++]) % 16) + 4;
        int64_t width = (static_cast<int64_t>(Data[offset++]) % 16) + 4;
        int64_t kernel_h = (static_cast<int64_t>(Data[offset++]) % 3) + 1;
        int64_t kernel_w = (static_cast<int64_t>(Data[offset++]) % 3) + 1;

        // Parse stride first (needed for output_padding constraint)
        int64_t stride_h = (static_cast<int64_t>(Data[offset++]) % 3) + 1;
        int64_t stride_w = (static_cast<int64_t>(Data[offset++]) % 3) + 1;
        std::vector<int64_t> stride = {stride_h, stride_w};

        // Parse padding
        int64_t pad_h = static_cast<int64_t>(Data[offset++]) % 4;
        int64_t pad_w = static_cast<int64_t>(Data[offset++]) % 4;
        std::vector<int64_t> padding = {pad_h, pad_w};

        // Output padding must be < stride
        int64_t out_pad_h = static_cast<int64_t>(Data[offset++]) % stride_h;
        int64_t out_pad_w = static_cast<int64_t>(Data[offset++]) % stride_w;
        std::vector<int64_t> output_padding = {out_pad_h, out_pad_w};

        // Parse dilation
        int64_t dilation_h = (static_cast<int64_t>(Data[offset++]) % 2) + 1;
        int64_t dilation_w = (static_cast<int64_t>(Data[offset++]) % 2) + 1;
        std::vector<int64_t> dilation = {dilation_h, dilation_w};

        // Parse groups - must divide both in_channels and out_channels
        int64_t groups = 1;
        if (offset < Size) {
            int64_t requested_groups = (static_cast<int64_t>(Data[offset++]) % 4) + 1;
            // Find valid groups value
            for (int64_t g = requested_groups; g >= 1; g--) {
                if (in_channels % g == 0 && out_channels % g == 0) {
                    groups = g;
                    break;
                }
            }
        }

        // Parse boolean flags
        bool benchmark = false;
        bool deterministic = false;
        bool allow_tf32 = false;
        if (offset < Size) {
            uint8_t flags = Data[offset++];
            benchmark = (flags & 0x01) != 0;
            deterministic = (flags & 0x02) != 0;
            allow_tf32 = (flags & 0x04) != 0;
        }

        // Create input tensor with shape [N, C_in, H, W]
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width}, options);

        // Create weight tensor with shape [C_in, C_out/groups, kH, kW] for transposed conv
        torch::Tensor weight = torch::randn({in_channels, out_channels / groups, kernel_h, kernel_w}, options);

        // Call cudnn_convolution_transpose
        try {
            torch::Tensor output = torch::cudnn_convolution_transpose(
                input, weight, padding, output_padding, stride, dilation,
                groups, benchmark, deterministic, allow_tf32
            );
            
            // Force computation
            output.sum().item<float>();
        } catch (const c10::Error &e) {
            // Expected errors from invalid parameter combinations
            return 0;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}