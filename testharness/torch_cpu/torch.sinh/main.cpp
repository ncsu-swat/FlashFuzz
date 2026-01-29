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
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // Apply sinh operation - basic variant
        torch::Tensor result = torch::sinh(input);

        // Use remaining data for options
        if (offset < Size) {
            uint8_t option_byte = Data[offset++];

            // Try out variant
            if (option_byte & 0x01) {
                torch::Tensor out = torch::empty_like(input);
                torch::sinh_out(out, input);
            }

            // Try in-place variant if tensor type supports it (floating point or complex)
            if ((option_byte & 0x02) && (input.is_floating_point() || input.is_complex())) {
                torch::Tensor input_copy = input.clone();
                input_copy.sinh_();
            }

            // Try with different dtypes
            if (option_byte & 0x04) {
                try {
                    torch::Tensor result_float = torch::sinh(input.to(torch::kFloat32));
                    (void)result_float;
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            }

            if (option_byte & 0x08) {
                try {
                    torch::Tensor result_double = torch::sinh(input.to(torch::kFloat64));
                    (void)result_double;
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            }

            if (option_byte & 0x10) {
                try {
                    // kHalf may not be supported on all CPU backends
                    torch::Tensor result_half = torch::sinh(input.to(torch::kFloat16));
                    (void)result_half;
                } catch (...) {
                    // Silently ignore - half precision may not be supported
                }
            }

            // Try with complex dtype
            if (option_byte & 0x20) {
                try {
                    torch::Tensor result_complex = torch::sinh(input.to(torch::kComplexFloat));
                    (void)result_complex;
                } catch (...) {
                    // Silently ignore complex conversion failures
                }
            }
        }

        // Create a second tensor and test sinh on it for more coverage
        if (offset + 2 < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor result2 = torch::sinh(input2);
            (void)result2;
        }

        // Prevent compiler from optimizing away results
        (void)result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}