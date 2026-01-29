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
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Apply torch.isreal operation
        torch::Tensor result = torch::isreal(input_tensor);

        // Verify result is defined and has correct properties
        if (result.defined()) {
            auto numel = result.numel();

            // Result should be boolean tensor with same shape as input
            auto sum = result.sum();
            auto all_true = result.all().item<bool>();
            auto any_true = result.any().item<bool>();

            // Access flattened values safely
            if (numel > 0) {
                auto flat = result.flatten();
                auto first_val = flat[0].item<bool>();
                if (numel > 1) {
                    auto last_val = flat[numel - 1].item<bool>();
                }
            }
        }

        // Test with explicitly created complex tensor if we have enough data
        if (offset + 4 < Size) {
            try {
                // Create a complex tensor to test isreal returns false
                size_t remaining_offset = 0;
                torch::Tensor real_part = fuzzer_utils::createTensor(Data + offset, Size - offset, remaining_offset);

                // Only proceed if we have a floating point tensor suitable for complex
                if (real_part.is_floating_point()) {
                    torch::Tensor imag_part = torch::ones_like(real_part);
                    torch::Tensor complex_tensor = torch::complex(real_part, imag_part);
                    torch::Tensor complex_result = torch::isreal(complex_tensor);

                    // Complex tensor with non-zero imaginary should have some false values
                    if (complex_result.defined()) {
                        auto any_real = complex_result.any().item<bool>();
                    }
                }
            }
            catch (...) {
                // Silently ignore failures in complex tensor creation
            }
        }

        // Test with known real tensor types
        if (Size > 4) {
            try {
                // Integer tensors are always real
                torch::Tensor int_tensor = torch::tensor({Data[0], Data[1], Data[2]}, torch::kInt32);
                torch::Tensor int_result = torch::isreal(int_tensor);
                auto all_real = int_result.all().item<bool>();

                // Float tensors (non-complex) are always real
                torch::Tensor float_tensor = torch::tensor({static_cast<float>(Data[0]), static_cast<float>(Data[1])});
                torch::Tensor float_result = torch::isreal(float_tensor);
            }
            catch (...) {
                // Silently ignore
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