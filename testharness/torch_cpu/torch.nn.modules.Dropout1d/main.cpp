#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters for Dropout
        float p = 0.5f; // Default dropout probability
        bool inplace = false;

        // Use first bytes for dropout probability
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);

            // Ensure p is in valid range [0, 1]
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5f;
            } else {
                p = std::abs(p);
                p = p - std::floor(p); // Get fractional part to ensure 0 <= p < 1
            }
        }

        // Use next byte for inplace parameter
        if (offset < Size) {
            inplace = (Data[offset++] & 0x01) != 0;
        }

        // Create input tensor - For 1D dropout, use 2D (N, C) or 3D (N, C, L) input
        size_t remaining = Size - offset;
        if (remaining < 4) {
            return 0;
        }

        // Determine dimensions from fuzzer data
        int64_t batch_size = 1 + (Data[offset] % 4);      // 1-4
        int64_t channels = 1 + (Data[offset + 1] % 8);    // 1-8
        int64_t length = 1 + (Data[offset + 2] % 16);     // 1-16
        offset += 3;

        // Create a 3D tensor (N, C, L) for 1D dropout testing
        torch::Tensor input = torch::randn({batch_size, channels, length});

        // Use torch::nn::Dropout which works for any dimensional input
        // In PyTorch C++, Dropout1d is not separately exposed, but Dropout works
        auto dropout = torch::nn::Dropout(
            torch::nn::DropoutOptions().p(p).inplace(inplace));

        // Test in evaluation mode (dropout should be disabled)
        dropout->eval();
        try {
            if (inplace) {
                torch::Tensor input_copy = input.clone();
                auto output_eval = dropout->forward(input_copy);
            } else {
                auto output_eval = dropout->forward(input);
            }
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors
        }

        // Test in training mode (dropout should be active)
        dropout->train();
        try {
            if (inplace) {
                torch::Tensor input_copy = input.clone();
                auto output_train = dropout->forward(input_copy);
            } else {
                auto output_train = dropout->forward(input);
            }
        } catch (const c10::Error&) {
            // Expected errors
        }

        // Test with zero dropout probability (should pass through)
        auto zero_dropout = torch::nn::Dropout(
            torch::nn::DropoutOptions().p(0.0).inplace(false));
        zero_dropout->train();
        try {
            auto output_zero = zero_dropout->forward(input);
        } catch (const c10::Error&) {
            // Expected errors
        }

        // Test with high dropout probability
        auto high_dropout = torch::nn::Dropout(
            torch::nn::DropoutOptions().p(0.9).inplace(false));
        high_dropout->train();
        try {
            auto output_high = high_dropout->forward(input);
        } catch (const c10::Error&) {
            // Expected errors
        }

        // Test with 2D input (N, C) - also valid for dropout
        torch::Tensor input_2d = torch::randn({batch_size, channels});
        try {
            auto output_2d = dropout->forward(input_2d);
        } catch (const c10::Error&) {
            // May not be supported depending on version
        }

        // Test functional dropout API which provides more control
        try {
            auto output_func = torch::nn::functional::dropout(
                input,
                torch::nn::functional::DropoutFuncOptions().p(p).training(true).inplace(false));
        } catch (const c10::Error&) {
            // Expected errors
        }

        // Test with different dtypes
        if (offset < Size && (Data[offset] & 0x01)) {
            try {
                torch::Tensor input_float64 = input.to(torch::kFloat64);
                auto output_f64 = dropout->forward(input_float64);
            } catch (const c10::Error&) {
                // Expected for unsupported dtypes
            }
        }

        // Test 1D specific behavior with feature dropout using functional API
        // This zeros out entire channels (features) similar to Dropout1d in Python
        if (offset + 1 < Size && (Data[offset] & 0x02)) {
            try {
                // For feature/channel dropout behavior, we can use dropout on permuted tensor
                // or use the standard dropout which works element-wise
                torch::Tensor input_4d = torch::randn({batch_size, channels, length, 1});
                auto output_4d = dropout->forward(input_4d);
            } catch (const c10::Error&) {
                // Expected errors
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}