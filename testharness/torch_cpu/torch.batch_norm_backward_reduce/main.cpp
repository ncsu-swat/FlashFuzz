#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        // Need enough data for meaningful fuzzing
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Parse dimensions for batch norm (N, C, H, W format is common)
        uint8_t batch_size = (Data[offset++] % 8) + 1;      // 1-8
        uint8_t num_channels = (Data[offset++] % 16) + 1;   // 1-16
        uint8_t height = (Data[offset++] % 8) + 1;          // 1-8
        uint8_t width = (Data[offset++] % 8) + 1;           // 1-8

        // Parse gradient flags
        bool input_g = Data[offset++] & 1;
        bool weight_g = Data[offset++] & 1;
        bool bias_g = Data[offset++] & 1;

        // Parse whether to include weight
        bool has_weight = Data[offset++] & 1;

        // Choose dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size && (Data[offset++] & 1)) {
            dtype = torch::kFloat64;
        }

        auto options = torch::TensorOptions().dtype(dtype);

        // Create grad_out tensor (N, C, H, W)
        torch::Tensor grad_out = torch::randn({batch_size, num_channels, height, width}, options);

        // Create input tensor x with same shape as grad_out
        torch::Tensor x = torch::randn({batch_size, num_channels, height, width}, options);

        // Create mean tensor (1D, size = num_channels)
        torch::Tensor mean = torch::randn({num_channels}, options);

        // Create invstd tensor (1D, size = num_channels, must be positive)
        torch::Tensor invstd = torch::abs(torch::randn({num_channels}, options)) + 0.01f;

        // Create optional weight tensor (1D, size = num_channels)
        std::optional<torch::Tensor> weight = std::nullopt;
        if (has_weight) {
            weight = torch::randn({num_channels}, options);
        }

        // Use remaining data to modify tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t grad_out_numel = grad_out.numel();
            size_t bytes_to_use = std::min(remaining, grad_out_numel * sizeof(float));
            
            auto grad_out_accessor = grad_out.data_ptr<float>();
            for (size_t i = 0; i < bytes_to_use / sizeof(float) && offset + sizeof(float) <= Size; i++) {
                float val;
                std::memcpy(&val, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isfinite(val)) {
                    grad_out_accessor[i % grad_out_numel] = val;
                }
            }
        }

        // Inner try-catch for expected API failures (shape issues, etc.)
        try
        {
            // Call batch_norm_backward_reduce
            auto result = torch::batch_norm_backward_reduce(
                grad_out,
                x,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g
            );

            // Unpack the result tuple
            auto sum_dy = std::get<0>(result);
            auto sum_dy_xmu = std::get<1>(result);
            auto grad_weight = std::get<2>(result);
            auto grad_bias = std::get<3>(result);

            // Use results to prevent optimization
            volatile float sink = 0.0f;
            if (sum_dy.defined() && sum_dy.numel() > 0) {
                sink += sum_dy.abs().sum().item<float>();
            }
            if (sum_dy_xmu.defined() && sum_dy_xmu.numel() > 0) {
                sink += sum_dy_xmu.abs().sum().item<float>();
            }
            if (grad_weight.defined() && grad_weight.numel() > 0) {
                sink += grad_weight.abs().sum().item<float>();
            }
            if (grad_bias.defined() && grad_bias.numel() > 0) {
                sink += grad_bias.abs().sum().item<float>();
            }
            (void)sink;
        }
        catch (const c10::Error &e)
        {
            // Expected errors from invalid tensor configurations - silently ignore
        }
        catch (const std::runtime_error &e)
        {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}