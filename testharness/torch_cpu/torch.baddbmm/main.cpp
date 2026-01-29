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
        // Need enough data for dimensions and scalars
        if (Size < 8)
            return 0;

        size_t offset = 0;

        // Parse batch size, n, m, p dimensions from fuzzer data
        // Keep dimensions small to avoid memory issues
        int64_t batch = (Data[offset++] % 8) + 1;  // 1-8
        int64_t n = (Data[offset++] % 16) + 1;     // 1-16
        int64_t m = (Data[offset++] % 16) + 1;     // 1-16
        int64_t p = (Data[offset++] % 16) + 1;     // 1-16

        // Parse dtype
        auto dtype_options = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            switch (dtype_choice) {
                case 0: dtype_options = torch::kFloat32; break;
                case 1: dtype_options = torch::kFloat64; break;
                case 2: dtype_options = torch::kFloat16; break;
            }
        }

        // Create tensors with proper shapes for baddbmm
        // baddbmm: out = beta * input + alpha * (batch1 @ batch2)
        // batch1: (b, n, m), batch2: (b, m, p), input: (b, n, p)
        torch::Tensor input = torch::randn({batch, n, p}, dtype_options);
        torch::Tensor batch1 = torch::randn({batch, n, m}, dtype_options);
        torch::Tensor batch2 = torch::randn({batch, m, p}, dtype_options);

        // Parse beta and alpha values
        float beta = 1.0f;
        float alpha = 1.0f;

        if (offset + sizeof(float) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp to reasonable range to avoid numerical issues
            if (std::isnan(beta) || std::isinf(beta)) beta = 1.0f;
            beta = std::max(-100.0f, std::min(100.0f, beta));
        }

        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(alpha) || std::isinf(alpha)) alpha = 1.0f;
            alpha = std::max(-100.0f, std::min(100.0f, alpha));
        }

        torch::Tensor result;

        // Try different variants of baddbmm
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 4;
        }

        switch (variant) {
            case 0:
                // baddbmm with beta and alpha
                result = torch::baddbmm(input, batch1, batch2, 
                    torch::Scalar(beta), torch::Scalar(alpha));
                break;
            case 1:
                // baddbmm with only beta (alpha defaults to 1)
                result = torch::baddbmm(input, batch1, batch2, 
                    torch::Scalar(beta));
                break;
            case 2:
                // baddbmm with defaults
                result = torch::baddbmm(input, batch1, batch2);
                break;
            case 3:
                // out variant - preallocate output tensor
                {
                    torch::Tensor out = torch::empty({batch, n, p}, dtype_options);
                    result = torch::baddbmm_out(out, input, batch1, batch2,
                        torch::Scalar(beta), torch::Scalar(alpha));
                }
                break;
        }

        // Use the result to prevent optimization
        volatile float sum_val = result.sum().item<float>();
        (void)sum_val;

        // Try in-place version
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            try {
                auto input_clone = input.clone();
                input_clone.baddbmm_(batch1, batch2, 
                    torch::Scalar(beta), torch::Scalar(alpha));
                volatile float sum2 = input_clone.sum().item<float>();
                (void)sum2;
            }
            catch (const std::exception &) {
                // Silently ignore in-place failures (expected for some dtypes)
            }
        }

        // Test with broadcasting if we have more data
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            try {
                // Test with 2D input (should broadcast)
                torch::Tensor input_2d = torch::randn({n, p}, dtype_options);
                torch::Tensor broadcast_result = torch::baddbmm(
                    input_2d, batch1, batch2,
                    torch::Scalar(beta), torch::Scalar(alpha));
                volatile float sum3 = broadcast_result.sum().item<float>();
                (void)sum3;
            }
            catch (const std::exception &) {
                // Broadcasting may not always work, silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}