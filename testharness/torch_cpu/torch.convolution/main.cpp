#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Determine convolution dimensions (1D, 2D, or 3D)
        uint8_t conv_dims = (Data[offset++] % 3) + 1; // 1, 2, or 3

        // Parse basic parameters
        int64_t batch_size = (Data[offset++] % 4) + 1;      // 1-4
        int64_t in_channels = (Data[offset++] % 8) + 1;     // 1-8
        int64_t out_channels = (Data[offset++] % 8) + 1;    // 1-8
        int64_t groups = (Data[offset++] % 4) + 1;          // 1-4

        // Adjust channels to be divisible by groups
        in_channels = ((in_channels + groups - 1) / groups) * groups;
        out_channels = ((out_channels + groups - 1) / groups) * groups;

        // Parse transposed flag
        bool transposed = Data[offset++] % 2 == 0;

        // Parse spatial dimensions, stride, padding, dilation, output_padding
        std::vector<int64_t> input_spatial;
        std::vector<int64_t> kernel_spatial;
        std::vector<int64_t> stride;
        std::vector<int64_t> padding;
        std::vector<int64_t> dilation;
        std::vector<int64_t> output_padding;

        for (int i = 0; i < conv_dims && offset + 5 <= Size; i++) {
            int64_t spatial_size = (Data[offset++] % 16) + 4;  // 4-19
            int64_t kernel_size = (Data[offset++] % 5) + 1;    // 1-5
            int64_t s = (Data[offset++] % 3) + 1;              // 1-3
            int64_t p = Data[offset++] % 3;                     // 0-2
            int64_t d = (Data[offset++] % 2) + 1;              // 1-2

            input_spatial.push_back(spatial_size);
            kernel_spatial.push_back(kernel_size);
            stride.push_back(s);
            padding.push_back(p);
            dilation.push_back(d);

            // Output padding for transposed convolution (must be < stride)
            if (transposed) {
                output_padding.push_back(Data[offset < Size ? offset++ : 0] % s);
            } else {
                output_padding.push_back(0);
            }
        }

        // Ensure we have valid dimensions
        if (input_spatial.size() != static_cast<size_t>(conv_dims)) {
            return 0;
        }

        // Build input shape: [N, C, D1, D2, ...]
        std::vector<int64_t> input_shape = {batch_size, in_channels};
        for (int64_t dim : input_spatial) {
            input_shape.push_back(dim);
        }

        // Build weight shape:
        // For regular conv: [out_channels, in_channels/groups, K1, K2, ...]
        // For transposed conv: [in_channels, out_channels/groups, K1, K2, ...]
        std::vector<int64_t> weight_shape;
        if (transposed) {
            weight_shape = {in_channels, out_channels / groups};
        } else {
            weight_shape = {out_channels, in_channels / groups};
        }
        for (int64_t k : kernel_spatial) {
            weight_shape.push_back(k);
        }

        // Determine dtype from fuzzer data
        torch::Dtype dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            if (dtype_choice == 1) {
                dtype = torch::kFloat64;
            } else if (dtype_choice == 2) {
                dtype = torch::kFloat16;
            }
        }

        // Create input tensor
        torch::Tensor input = torch::randn(input_shape, torch::TensorOptions().dtype(dtype));

        // Create weight tensor
        torch::Tensor weight = torch::randn(weight_shape, torch::TensorOptions().dtype(dtype));

        // Create bias tensor (optional)
        c10::optional<torch::Tensor> bias = c10::nullopt;
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        if (use_bias) {
            bias = torch::randn({out_channels}, torch::TensorOptions().dtype(dtype));
        }

        // Apply convolution operation
        torch::Tensor output;
        try {
            output = torch::convolution(
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups
            );
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid combinations
            return 0;
        }

        // Perform operations on output to ensure it's used and computed
        if (output.defined()) {
            auto sum = output.sum();
            auto mean = output.mean();
            
            // Access the values to ensure computation happens
            volatile float sum_val = sum.item<float>();
            volatile float mean_val = mean.item<float>();
            (void)sum_val;
            (void)mean_val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}