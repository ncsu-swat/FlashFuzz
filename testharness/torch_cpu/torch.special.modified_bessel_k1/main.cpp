#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor - modified_bessel_k1 requires floating point input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }

        // modified_bessel_k1 is defined for non-negative real values
        // Use abs to ensure valid domain
        input = torch::abs(input);

        // Apply the modified_bessel_k1 operation
        torch::Tensor result;
        try {
            result = torch::special::modified_bessel_k1(input);
        } catch (const std::exception &) {
            // Expected failures (unsupported dtype, etc.)
            return 0;
        }

        // Access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }

        // Test with double precision if we have enough data
        if (offset + 2 < Size) {
            try {
                torch::Tensor input_double = input.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::modified_bessel_k1(input_double);
                if (result_double.defined() && result_double.numel() > 0) {
                    volatile double sum = result_double.sum().item<double>();
                    (void)sum;
                }
            } catch (const std::exception &) {
                // Silently handle expected failures
            }
        }

        // Try with out variant
        try {
            // Create output tensor with same shape and dtype as input
            torch::Tensor out = torch::empty_like(input);

            // Apply the operation with out parameter
            torch::special::modified_bessel_k1_out(out, input);

            // Access the result
            if (out.defined() && out.numel() > 0) {
                volatile float sum = out.sum().item<float>();
                (void)sum;
            }
        } catch (const std::exception &) {
            // Silently handle expected failures for out variant
        }

        // Test with different tensor shapes
        if (offset + 4 < Size) {
            try {
                // Create a 2D tensor
                int rows = (Data[offset % Size] % 8) + 1;
                int cols = (Data[(offset + 1) % Size] % 8) + 1;
                torch::Tensor input_2d = torch::rand({rows, cols}, torch::kFloat32);
                
                // Scale to reasonable range for bessel function
                input_2d = input_2d * 10.0f;
                
                torch::Tensor result_2d = torch::special::modified_bessel_k1(input_2d);
                if (result_2d.defined() && result_2d.numel() > 0) {
                    volatile float sum = result_2d.sum().item<float>();
                    (void)sum;
                }
            } catch (const std::exception &) {
                // Silently handle expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}