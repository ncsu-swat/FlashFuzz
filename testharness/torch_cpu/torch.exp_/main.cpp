#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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

        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Apply the exp_ operation in-place
        tensor.exp_();

        // Try with another tensor if we have remaining data
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            tensor2.exp_();
        }

        // Use fuzzer data to select additional test cases
        uint8_t test_selector = (Size > 0) ? Data[0] : 0;

        // Try with empty tensor
        if (test_selector & 0x01) {
            torch::Tensor empty_tensor = torch::empty({0});
            empty_tensor.exp_();
        }

        // Try with scalar tensor
        if (test_selector & 0x02) {
            float value = (Size > 1) ? static_cast<float>(Data[1]) / 255.0f : 0.5f;
            torch::Tensor scalar_tensor = torch::tensor(value);
            scalar_tensor.exp_();
        }

        // Try with tensors containing extreme values
        if (test_selector & 0x04) {
            std::vector<float> large_values = {1e30f, -1e30f, 1e-30f, -1e-30f};
            torch::Tensor large_tensor = torch::tensor(large_values);
            large_tensor.exp_();
        }

        // Try with special values (inf, -inf, nan)
        if (test_selector & 0x08) {
            std::vector<float> special_values = {
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f,
                -0.0f
            };
            torch::Tensor special_tensor = torch::tensor(special_values);
            special_tensor.exp_();
        }

        // Try with double precision tensor
        if (test_selector & 0x10) {
            torch::Tensor double_tensor = tensor.to(torch::kFloat64);
            double_tensor.exp_();
        }

        // Try with contiguous and non-contiguous tensors
        if ((test_selector & 0x20) && tensor.numel() > 1) {
            try {
                // Create a non-contiguous tensor via transpose if possible
                if (tensor.dim() >= 2 && tensor.size(0) > 1 && tensor.size(1) > 1) {
                    torch::Tensor transposed = tensor.transpose(0, 1);
                    transposed.exp_();
                }
            } catch (...) {
                // Silently ignore shape-related failures
            }
        }

        // Try with different dtypes
        if (test_selector & 0x40) {
            try {
                torch::Tensor half_tensor = tensor.to(torch::kFloat16);
                half_tensor.exp_();
            } catch (...) {
                // Half precision may not be supported on all systems
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