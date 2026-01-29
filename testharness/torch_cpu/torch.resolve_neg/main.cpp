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

        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Apply torch::resolve_neg operation
        torch::Tensor result = torch::resolve_neg(input_tensor);

        // Verify the result is defined and has expected properties
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            (void)sizes;
            (void)dtype;

            // Force evaluation by computing sum instead of item()
            if (result.numel() > 0 && result.is_floating_point()) {
                try {
                    auto sum = result.sum();
                    (void)sum;
                } catch (...) {
                    // Silently ignore computation errors
                }
            }
        }

        // Test with negated tensor - this creates a tensor with negative bit set
        {
            torch::Tensor neg_tensor = input_tensor.neg();
            torch::Tensor neg_result = torch::resolve_neg(neg_tensor);

            if (neg_result.defined() && neg_result.numel() > 0) {
                try {
                    auto sum = neg_result.sum();
                    (void)sum;
                } catch (...) {
                    // Silently ignore
                }
            }
        }

        // Test with _neg_view if available - this explicitly sets negative bit
        try {
            torch::Tensor neg_view = input_tensor._neg_view();
            torch::Tensor resolved = torch::resolve_neg(neg_view);

            if (resolved.defined() && resolved.numel() > 0) {
                auto sum = resolved.sum();
                (void)sum;
            }
        } catch (...) {
            // _neg_view might not be available for all tensor types
        }

        // Test with a slice (non-contiguous tensor)
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            try {
                torch::Tensor sliced = input_tensor.slice(0, 0, input_tensor.size(0) / 2 + 1);
                torch::Tensor sliced_result = torch::resolve_neg(sliced);
                
                if (sliced_result.defined() && sliced_result.numel() > 0) {
                    auto sum = sliced_result.sum();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore slicing errors
            }
        }

        // Test with complex tensor if we have enough data
        if (offset + 4 < Size) {
            try {
                // Create a small complex tensor
                int dim = (Data[offset % Size] % 4) + 1;
                torch::Tensor real_part = torch::randn({dim, dim});
                torch::Tensor imag_part = torch::randn({dim, dim});
                torch::Tensor complex_tensor = torch::complex(real_part, imag_part);

                // Negate and resolve
                torch::Tensor neg_complex = complex_tensor.neg();
                torch::Tensor resolved_complex = torch::resolve_neg(neg_complex);

                if (resolved_complex.defined() && resolved_complex.numel() > 0) {
                    auto sum = resolved_complex.sum();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore complex tensor errors
            }
        }

        // Test with scalar tensor
        {
            try {
                float scalar_val = static_cast<float>(Data[0] % 256) - 128.0f;
                torch::Tensor scalar_tensor = torch::tensor(scalar_val);
                torch::Tensor scalar_neg = scalar_tensor.neg();
                torch::Tensor scalar_result = torch::resolve_neg(scalar_neg);

                if (scalar_result.defined()) {
                    auto val = scalar_result.item<float>();
                    (void)val;
                }
            } catch (...) {
                // Silently ignore scalar errors
            }
        }

        // Test with zero tensor
        {
            try {
                torch::Tensor zero_tensor = torch::zeros_like(input_tensor);
                torch::Tensor zero_neg = zero_tensor.neg();
                torch::Tensor zero_result = torch::resolve_neg(zero_neg);

                if (zero_result.defined() && zero_result.numel() > 0) {
                    auto sum = zero_result.sum();
                    (void)sum;
                }
            } catch (...) {
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