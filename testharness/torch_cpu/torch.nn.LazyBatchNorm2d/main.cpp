#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cstdint>        // For uint64_t

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
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for BatchNorm2d from the data
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp eps to reasonable range to avoid numerical issues
            if (std::isnan(eps) || std::isinf(eps) || eps <= 0) {
                eps = 1e-5;
            }
            eps = std::max(1e-10, std::min(eps, 1.0));
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp momentum to valid range [0, 1]
            if (std::isnan(momentum) || std::isinf(momentum)) {
                momentum = 0.1;
            }
            momentum = std::max(0.0, std::min(momentum, 1.0));
        }
        
        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Create 4D input tensor for BatchNorm2d (N, C, H, W)
        // Extract dimensions from remaining data
        int64_t batch_size = 1;
        int64_t channels = 3;
        int64_t height = 4;
        int64_t width = 4;
        
        if (offset + 4 <= Size) {
            batch_size = std::max(int64_t(1), static_cast<int64_t>((Data[offset++] % 8) + 1));
            channels = std::max(int64_t(1), static_cast<int64_t>((Data[offset++] % 32) + 1));
            height = std::max(int64_t(1), static_cast<int64_t>((Data[offset++] % 16) + 1));
            width = std::max(int64_t(1), static_cast<int64_t>((Data[offset++] % 16) + 1));
        }
        
        // Create input tensor with proper 4D shape
        torch::Tensor input = torch::randn({batch_size, channels, height, width});
        
        // If we have more data, use it to modify the tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            auto input_accessor = input.accessor<float, 4>();
            size_t idx = 0;
            for (int64_t n = 0; n < batch_size && idx < remaining; ++n) {
                for (int64_t c = 0; c < channels && idx < remaining; ++c) {
                    for (int64_t h = 0; h < height && idx < remaining; ++h) {
                        for (int64_t w = 0; w < width && idx < remaining; ++w) {
                            input_accessor[n][c][h][w] = static_cast<float>(Data[offset + idx]) / 128.0f - 1.0f;
                            ++idx;
                        }
                    }
                }
            }
        }
        
        // Create BatchNorm2d module with num_features = channels
        // Note: LazyBatchNorm2d is Python-only; C++ requires explicit num_features
        auto batch_norm = torch::nn::BatchNorm2d(
            torch::nn::BatchNorm2dOptions(channels)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the batch norm operation in training mode
        batch_norm->train();
        torch::Tensor output;
        try {
            output = batch_norm->forward(input);
        } catch (const c10::Error&) {
            // Shape mismatches or other expected errors
            return 0;
        }
        
        // Force computation to ensure any errors are triggered
        output = output.contiguous();
        
        // Access some elements to ensure computation happens
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
        
        // Test eval mode as well
        batch_norm->eval();
        try {
            torch::Tensor output_eval = batch_norm->forward(input);
            output_eval = output_eval.contiguous();
            if (output_eval.numel() > 0) {
                float sum_eval = output_eval.sum().item<float>();
                (void)sum_eval;
            }
        } catch (const c10::Error&) {
            // May fail if running stats weren't tracked
        }
        
        // Test with different spatial dimensions but same channel count
        if (offset + 2 <= Size) {
            int64_t new_height = std::max(int64_t(1), static_cast<int64_t>((Data[Size - 2] % 8) + 1));
            int64_t new_width = std::max(int64_t(1), static_cast<int64_t>((Data[Size - 1] % 8) + 1));
            torch::Tensor input2 = torch::randn({batch_size, channels, new_height, new_width});
            
            batch_norm->train();
            try {
                torch::Tensor output2 = batch_norm->forward(input2);
                (void)output2.sum().item<float>();
            } catch (const c10::Error&) {
                // Different spatial dimensions should still work
            }
        }
        
        // Test with different batch sizes
        if (Size > 4) {
            int64_t new_batch = std::max(int64_t(1), static_cast<int64_t>((Data[Size / 2] % 4) + 1));
            torch::Tensor input3 = torch::randn({new_batch, channels, height, width});
            
            batch_norm->train();
            try {
                torch::Tensor output3 = batch_norm->forward(input3);
                (void)output3.sum().item<float>();
            } catch (const c10::Error&) {
                // Expected for edge cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}