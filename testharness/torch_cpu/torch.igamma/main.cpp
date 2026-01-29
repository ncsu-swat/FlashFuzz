#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
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
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create the first tensor (a) - should be positive for igamma
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have data for second tensor
        if (offset >= Size) {
            return 0;
        }

        // Create the second tensor (x) - should be non-negative for igamma
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);

        // Convert to floating point if needed (igamma requires float/double)
        if (!a.is_floating_point()) {
            a = a.to(torch::kFloat);
        }
        if (!x.is_floating_point()) {
            x = x.to(torch::kFloat);
        }

        // Test 1: Basic igamma with tensor inputs
        try {
            torch::Tensor result = torch::igamma(a, x);
            (void)result;
        }
        catch (const std::exception&) {
            // Shape mismatch or other expected failure, ignore
        }

        // Test 2: igamma with abs to ensure valid domain (a > 0, x >= 0)
        try {
            torch::Tensor a_pos = torch::abs(a) + 0.001f;  // Ensure positive
            torch::Tensor x_nonneg = torch::abs(x);        // Ensure non-negative
            torch::Tensor result = torch::igamma(a_pos, x_nonneg);
            (void)result;
        }
        catch (const std::exception&) {
            // Expected failure, ignore
        }

        // Test 3: Try with scalar values from raw data
        if (offset + 16 <= Size) {
            double a_scalar, x_scalar;
            std::memcpy(&a_scalar, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&x_scalar, Data + offset, sizeof(double));
            offset += sizeof(double);

            // Skip invalid floating point values
            if (std::isfinite(a_scalar) && std::isfinite(x_scalar)) {
                try {
                    auto a_tensor = torch::tensor(std::abs(a_scalar) + 0.001);
                    auto x_tensor = torch::tensor(std::abs(x_scalar));
                    torch::Tensor result = torch::igamma(a_tensor, x_tensor);
                    (void)result;
                }
                catch (const std::exception&) {
                    // Expected failure, ignore
                }

                // Test scalar broadcast with tensor
                try {
                    auto a_scalar_tensor = torch::tensor(std::abs(a_scalar) + 0.001);
                    torch::Tensor result = torch::igamma(a_scalar_tensor, torch::abs(x));
                    (void)result;
                }
                catch (const std::exception&) {
                    // Expected failure, ignore
                }

                try {
                    auto x_scalar_tensor = torch::tensor(std::abs(x_scalar));
                    torch::Tensor result = torch::igamma(torch::abs(a) + 0.001f, x_scalar_tensor);
                    (void)result;
                }
                catch (const std::exception&) {
                    // Expected failure, ignore
                }
            }
        }

        // Test 4: Test with different dtypes
        try {
            torch::Tensor a_double = a.to(torch::kDouble);
            torch::Tensor x_double = x.to(torch::kDouble);
            torch::Tensor result = torch::igamma(torch::abs(a_double) + 0.001, torch::abs(x_double));
            (void)result;
        }
        catch (const std::exception&) {
            // Expected failure, ignore
        }

        // Test 5: Test with contiguous/non-contiguous tensors
        try {
            if (a.dim() >= 2 && x.dim() >= 2) {
                torch::Tensor a_t = a.transpose(0, 1);
                torch::Tensor x_t = x.transpose(0, 1);
                torch::Tensor result = torch::igamma(torch::abs(a_t) + 0.001f, torch::abs(x_t));
                (void)result;
            }
        }
        catch (const std::exception&) {
            // Expected failure, ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}