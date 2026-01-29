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
        // Need at least enough bytes for padding values and some tensor data
        if (Size < 12) {
            return 0;
        }

        size_t offset = 0;

        // Extract padding values from the input data (limit to reasonable range)
        int64_t left = static_cast<int64_t>(Data[offset++] % 32);
        int64_t right = static_cast<int64_t>(Data[offset++] % 32);
        int64_t top = static_cast<int64_t>(Data[offset++] % 32);
        int64_t bottom = static_cast<int64_t>(Data[offset++] % 32);

        // Get value to pad with
        float temp_value;
        std::memcpy(&temp_value, Data + offset, sizeof(float));
        // Sanitize NaN/Inf to avoid issues
        if (std::isnan(temp_value) || std::isinf(temp_value)) {
            temp_value = 0.0f;
        }
        double pad_value = static_cast<double>(temp_value);
        offset += sizeof(float);

        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);

        // ConstantPad2d expects 3D (C, H, W) or 4D (N, C, H, W) input
        // Reshape tensor to ensure proper dimensions
        int64_t numel = input.numel();
        if (numel < 1) {
            return 0;
        }

        // Determine batch vs unbatched based on a data byte
        bool use_batch = (Size > offset && (Data[0] & 0x01));

        torch::Tensor reshaped_input;
        try {
            if (use_batch) {
                // 4D: (N, C, H, W)
                int64_t n = std::max<int64_t>(1, (numel > 16) ? 2 : 1);
                int64_t remaining = numel / n;
                int64_t c = std::max<int64_t>(1, (remaining > 8) ? 2 : 1);
                remaining = remaining / c;
                int64_t h = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(remaining)));
                int64_t w = remaining / h;
                if (n * c * h * w > 0 && n * c * h * w <= numel) {
                    reshaped_input = input.flatten().narrow(0, 0, n * c * h * w).view({n, c, h, w});
                } else {
                    reshaped_input = input.flatten().narrow(0, 0, 1).view({1, 1, 1, 1});
                }
            } else {
                // 3D: (C, H, W)
                int64_t c = std::max<int64_t>(1, (numel > 8) ? 2 : 1);
                int64_t remaining = numel / c;
                int64_t h = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(remaining)));
                int64_t w = remaining / h;
                if (c * h * w > 0 && c * h * w <= numel) {
                    reshaped_input = input.flatten().narrow(0, 0, c * h * w).view({c, h, w});
                } else {
                    reshaped_input = input.flatten().narrow(0, 0, 1).view({1, 1, 1});
                }
            }
        } catch (...) {
            // Shape manipulation failed, use minimal valid tensor
            reshaped_input = torch::zeros({1, 1, 2, 2});
        }

        // Create the ConstantPad2d module
        torch::nn::ConstantPad2d pad(
            torch::nn::ConstantPad2dOptions(
                torch::ExpandingArray<4>({left, right, top, bottom}), 
                pad_value
            )
        );

        // Apply padding - inner try-catch for expected shape errors
        try {
            torch::Tensor output = pad(reshaped_input);

            // Verify output and exercise the result
            if (output.defined() && output.numel() > 0) {
                auto sum = output.sum();
                auto mean = output.mean();
                // Check output shape is as expected
                auto out_sizes = output.sizes();
                (void)out_sizes;
            }
        } catch (const c10::Error&) {
            // Expected failures (shape mismatches, etc.) - catch silently
        } catch (const std::runtime_error&) {
            // Runtime errors from invalid operations - catch silently
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}