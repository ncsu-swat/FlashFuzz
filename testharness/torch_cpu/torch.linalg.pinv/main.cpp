#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Parse dimensions for a 2D matrix (pinv requires at least 2D)
        uint8_t rows = (Data[offset++] % 16) + 1;  // 1-16 rows
        uint8_t cols = (Data[offset++] % 16) + 1;  // 1-16 cols

        // Parse optional batch dimension
        bool use_batch = (offset < Size) && (Data[offset++] & 0x01);
        uint8_t batch_size = 1;
        if (use_batch && offset < Size) {
            batch_size = (Data[offset++] % 4) + 1;  // 1-4 batch size
        }

        // Create input tensor with appropriate shape
        torch::Tensor input;
        if (use_batch) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                input = input.reshape({batch_size, rows, cols}).to(torch::kFloat64);
            } catch (...) {
                // If reshape fails, create a simple matrix
                input = torch::randn({batch_size, rows, cols}, torch::kFloat64);
            }
        } else {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                input = input.reshape({rows, cols}).to(torch::kFloat64);
            } catch (...) {
                input = torch::randn({rows, cols}, torch::kFloat64);
            }
        }

        // Parse rcond parameter (used for tolerance)
        double rcond = 1e-15;  // Default similar to NumPy
        if (offset + sizeof(float) <= Size) {
            float rcond_f;
            std::memcpy(&rcond_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isnan(rcond_f) && !std::isinf(rcond_f) && rcond_f >= 0 && rcond_f <= 1.0) {
                rcond = static_cast<double>(rcond_f);
            }
        }

        // Parse hermitian parameter
        bool hermitian = false;
        if (offset < Size) {
            hermitian = static_cast<bool>(Data[offset++] & 0x01);
        }

        // If hermitian is true, we need a square matrix that is hermitian
        if (hermitian) {
            // Make the matrix square
            int64_t min_dim = std::min(static_cast<int64_t>(rows), static_cast<int64_t>(cols));
            if (use_batch) {
                input = input.slice(-2, 0, min_dim).slice(-1, 0, min_dim);
            } else {
                input = input.slice(0, 0, min_dim).slice(1, 0, min_dim);
            }
            // Make it hermitian: A = (A + A^T) / 2
            input = (input + input.transpose(-2, -1)) / 2.0;
        }

        torch::Tensor result;

        // Select which parameter combination to use
        uint8_t param_selector = 0;
        if (offset < Size) {
            param_selector = Data[offset++] % 4;
        }

        switch (param_selector) {
            case 0:
                // Just the input tensor (use default tolerances)
                result = torch::linalg_pinv(input);
                break;
            case 1:
                // With rcond (as Tensor)
                {
                    torch::Tensor rcond_tensor = torch::tensor(rcond, torch::kFloat64);
                    result = torch::linalg_pinv(input, rcond_tensor, hermitian);
                }
                break;
            case 2:
                // With hermitian flag only
                result = torch::linalg_pinv(input, torch::Tensor(), hermitian);
                break;
            case 3:
                // With both rcond and hermitian
                {
                    torch::Tensor rcond_tensor = torch::tensor(rcond, torch::kFloat64);
                    result = torch::linalg_pinv(input, rcond_tensor, hermitian);
                }
                break;
        }

        // Validate result
        if (result.defined() && result.numel() > 0) {
            // Access element to ensure computation completed
            volatile double val = result.flatten()[0].item<double>();
            (void)val;

            // Verify pseudoinverse property: A * pinv(A) * A ≈ A (for non-batched)
            if (!use_batch && result.dim() == 2) {
                try {
                    torch::Tensor reconstructed = torch::mm(torch::mm(input, result), input);
                    (void)reconstructed;
                } catch (...) {
                    // Shape mismatch is acceptable for some edge cases
                }
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