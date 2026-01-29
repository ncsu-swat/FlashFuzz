#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values first (constrain to reasonable range)
        int64_t padding_left = static_cast<int64_t>(Data[offset++] % 32);
        int64_t padding_right = static_cast<int64_t>(Data[offset++] % 32);
        
        // Extract dimensions for a proper 3D tensor (N, C, W)
        int64_t batch_size = static_cast<int64_t>((Data[offset++] % 4) + 1);
        int64_t channels = static_cast<int64_t>((Data[offset++] % 8) + 1);
        // Width must be at least max(padding_left, padding_right) for circular padding
        int64_t min_width = std::max(padding_left, padding_right);
        int64_t width = static_cast<int64_t>((Data[offset++] % 32) + 1) + min_width;
        
        // Create a 3D input tensor suitable for circular padding (N, C, W)
        torch::Tensor input = torch::randn({batch_size, channels, width});
        
        // Use remaining data to modify tensor values if available
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t tensor_size = static_cast<size_t>(input.numel());
            size_t copy_size = std::min(remaining, tensor_size * sizeof(float));
            
            if (copy_size >= sizeof(float)) {
                auto accessor = input.accessor<float, 3>();
                size_t idx = 0;
                for (int64_t b = 0; b < batch_size && (idx + 1) * sizeof(float) <= copy_size; b++) {
                    for (int64_t c = 0; c < channels && (idx + 1) * sizeof(float) <= copy_size; c++) {
                        for (int64_t w = 0; w < width && (idx + 1) * sizeof(float) <= copy_size; w++) {
                            float val;
                            std::memcpy(&val, Data + offset + idx * sizeof(float), sizeof(float));
                            if (std::isfinite(val)) {
                                accessor[b][c][w] = val;
                            }
                            idx++;
                        }
                    }
                }
            }
        }
        
        // Apply circular padding using functional API
        // For 1D padding on 3D input, pad format is (left, right)
        try {
            torch::Tensor output = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions({padding_left, padding_right})
                    .mode(torch::kCircular)
            );
            
            // Use the output to prevent optimization
            if (output.defined()) {
                volatile auto sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected failures for invalid padding configurations
        }
        
        // Also test with 2D input (unbatched: C, W)
        // Width must still be large enough for circular padding
        torch::Tensor input_2d = torch::randn({channels, width});
        try {
            torch::Tensor output_2d = torch::nn::functional::pad(
                input_2d,
                torch::nn::functional::PadFuncOptions({padding_left, padding_right})
                    .mode(torch::kCircular)
            );
            if (output_2d.defined()) {
                volatile auto sum = output_2d.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected failures
        }
        
        // Test with symmetric padding (same padding on both sides)
        try {
            int64_t symmetric_pad = (padding_left + padding_right) / 2;
            if (symmetric_pad > 0 && symmetric_pad <= width) {
                torch::Tensor output_sym = torch::nn::functional::pad(
                    input,
                    torch::nn::functional::PadFuncOptions({symmetric_pad, symmetric_pad})
                        .mode(torch::kCircular)
                );
                if (output_sym.defined()) {
                    volatile auto sum = output_sym.sum().item<float>();
                    (void)sum;
                }
            }
        } catch (const c10::Error&) {
            // Expected failures
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}