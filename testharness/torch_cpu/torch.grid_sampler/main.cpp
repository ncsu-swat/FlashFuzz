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
        // Need enough data to parse parameters
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Parse dimensions from fuzzer data for controlled tensor creation
        int64_t batch_size = static_cast<int64_t>((Data[offset++] % 4) + 1);  // 1-4
        int64_t channels = static_cast<int64_t>((Data[offset++] % 4) + 1);    // 1-4
        int64_t in_height = static_cast<int64_t>((Data[offset++] % 8) + 1);   // 1-8
        int64_t in_width = static_cast<int64_t>((Data[offset++] % 8) + 1);    // 1-8
        int64_t out_height = static_cast<int64_t>((Data[offset++] % 8) + 1);  // 1-8
        int64_t out_width = static_cast<int64_t>((Data[offset++] % 8) + 1);   // 1-8

        // Parse interpolation mode: 0=bilinear, 1=nearest, 2=bicubic
        int64_t interpolation_mode = static_cast<int64_t>(Data[offset++]) % 3;

        // Parse padding mode: 0=zeros, 1=border, 2=reflection
        int64_t padding_mode = static_cast<int64_t>(Data[offset++]) % 3;

        // Parse align_corners flag
        bool align_corners = false;
        if (offset < Size) {
            align_corners = static_cast<bool>(Data[offset++] & 0x01);
        }

        // Bicubic interpolation requires align_corners=True for some PyTorch versions
        // Also bicubic doesn't support all padding modes in some versions
        if (interpolation_mode == 2) {
            align_corners = true;
        }

        // Create input tensor: (N, C, H_in, W_in)
        torch::Tensor input = torch::randn({batch_size, channels, in_height, in_width});

        // Create grid tensor: (N, H_out, W_out, 2)
        // Grid values should be in range [-1, 1] for normalized coordinates
        torch::Tensor grid = torch::rand({batch_size, out_height, out_width, 2}) * 2.0 - 1.0;

        // Optionally add some out-of-range values to test padding modes
        if (offset < Size && (Data[offset++] & 0x01)) {
            // Add some values outside [-1, 1] to exercise padding
            auto mask = torch::rand({batch_size, out_height, out_width, 2}) > 0.8;
            auto extreme = (torch::rand({batch_size, out_height, out_width, 2}) * 4.0 - 2.0);
            grid = torch::where(mask, extreme, grid);
        }

        // Apply grid_sampler operation
        torch::Tensor output = torch::grid_sampler(
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners
        );

        // Verify output shape: should be (N, C, H_out, W_out)
        if (output.dim() != 4 ||
            output.size(0) != batch_size ||
            output.size(1) != channels ||
            output.size(2) != out_height ||
            output.size(3) != out_width) {
            std::cerr << "Unexpected output shape" << std::endl;
            return -1;
        }

        // Ensure the output is used to prevent optimization
        auto sum = output.sum().item<float>();
        (void)sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}