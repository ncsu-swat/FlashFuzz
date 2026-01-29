#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For INFINITY, NAN

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
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // airy_ai only supports floating point types (float, double)
        // Convert to float if not already a floating type
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }

        // Apply torch.special.airy_ai operation
        torch::Tensor result = torch::special::airy_ai(input);

        // Verify result is computed (access first element to force computation)
        if (result.defined() && result.numel() > 0) {
            volatile auto _ = result.index({0}).item<float>();
        }

        // Test with out parameter variant if we have more data
        if (offset + 2 < Size) {
            // Create output tensor with same shape and compatible dtype
            torch::Tensor output = torch::empty_like(result);
            torch::special::airy_ai_out(output, input);

            if (output.defined() && output.numel() > 0) {
                volatile auto _ = output.index({0}).item<float>();
            }
        }

        // Test with double precision
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor double_result = torch::special::airy_ai(double_input);
            if (double_result.defined() && double_result.numel() > 0) {
                volatile auto _ = double_result.index({0}).item<double>();
            }
        } catch (...) {
            // Silently ignore conversion/computation errors
        }

        // Test with different tensor shapes based on fuzzer data
        if (offset < Size) {
            uint8_t shape_selector = Data[offset++] % 4;
            torch::Tensor shaped_input;

            try {
                switch (shape_selector) {
                    case 0:
                        // Scalar
                        shaped_input = torch::tensor(input.flatten().index({0}).item<float>());
                        break;
                    case 1:
                        // 1D tensor
                        shaped_input = input.flatten();
                        break;
                    case 2:
                        // 2D tensor - reshape if possible
                        if (input.numel() >= 4) {
                            int64_t n = static_cast<int64_t>(std::sqrt(input.numel()));
                            if (n > 0) {
                                shaped_input = input.flatten().narrow(0, 0, n * n).reshape({n, n});
                            }
                        }
                        break;
                    case 3:
                        // Contiguous view
                        shaped_input = input.contiguous();
                        break;
                }

                if (shaped_input.defined() && shaped_input.numel() > 0) {
                    torch::Tensor shaped_result = torch::special::airy_ai(shaped_input);
                }
            } catch (...) {
                // Silently ignore shape-related errors
            }
        }

        // Test with special values
        if (offset < Size) {
            uint8_t special_selector = Data[offset++] % 5;
            torch::Tensor special_input;

            try {
                switch (special_selector) {
                    case 0:
                        // Large positive values
                        special_input = torch::ones({2, 2}, torch::kFloat) * 100.0f;
                        break;
                    case 1:
                        // Large negative values (where Airy Ai oscillates)
                        special_input = torch::ones({2, 2}, torch::kFloat) * -100.0f;
                        break;
                    case 2:
                        // Values near zero
                        special_input = torch::zeros({2, 2}, torch::kFloat) + 1e-10f;
                        break;
                    case 3:
                        // Special floating point values
                        special_input = torch::tensor({INFINITY, -INFINITY, NAN, 0.0f}, torch::kFloat);
                        break;
                    case 4:
                        // Mix of positive and negative
                        special_input = torch::linspace(-10.0f, 10.0f, 10);
                        break;
                }

                if (special_input.defined()) {
                    torch::Tensor special_result = torch::special::airy_ai(special_input);
                }
            } catch (...) {
                // Silently ignore special value errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}