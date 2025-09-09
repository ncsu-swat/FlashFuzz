#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        int64_t window_length_raw;
        std::memcpy(&window_length_raw, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        int64_t window_length = std::abs(window_length_raw) % 10000;
        if (window_length == 0) {
            window_length = 1;
        }

        bool periodic = (Data[offset] % 2) == 1;
        offset++;

        torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset]);
        offset++;

        bool requires_grad = (Data[offset] % 2) == 1;
        offset++;

        auto options = torch::TensorOptions()
            .dtype(dtype)
            .requires_grad(requires_grad);

        torch::Tensor result = torch::hann_window(window_length, periodic, options);

        torch::Tensor result2 = torch::hann_window(window_length, !periodic, options);

        if (window_length > 1) {
            torch::Tensor result3 = torch::hann_window(1, periodic, options);
        }

        if (window_length < 5000) {
            torch::Tensor result4 = torch::hann_window(window_length * 2, periodic, options);
        }

        torch::Tensor large_window = torch::hann_window(65536, periodic, options);

        torch::Tensor small_window = torch::hann_window(2, periodic, options);

        auto float_options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor float_result = torch::hann_window(window_length, periodic, float_options);

        auto double_options = torch::TensorOptions().dtype(torch::kDouble);
        torch::Tensor double_result = torch::hann_window(window_length, periodic, double_options);

        if (torch::cuda::is_available()) {
            auto cuda_options = torch::TensorOptions().device(torch::kCUDA);
            torch::Tensor cuda_result = torch::hann_window(window_length, periodic, cuda_options);
        }

        torch::Tensor edge_case1 = torch::hann_window(1, true);
        torch::Tensor edge_case2 = torch::hann_window(1, false);

        if (window_length > 2) {
            torch::Tensor periodic_test = torch::hann_window(window_length, true);
            torch::Tensor symmetric_test = torch::hann_window(window_length + 1, false);
            if (symmetric_test.size(0) > 0) {
                torch::Tensor sliced = symmetric_test.slice(0, 0, -1);
            }
        }

        auto half_options = torch::TensorOptions().dtype(torch::kHalf);
        torch::Tensor half_result = torch::hann_window(window_length, periodic, half_options);

        auto bfloat16_options = torch::TensorOptions().dtype(torch::kBFloat16);
        torch::Tensor bfloat16_result = torch::hann_window(window_length, periodic, bfloat16_options);

        torch::Tensor very_large = torch::hann_window(100000, periodic);

        for (int i = 1; i <= 10; i++) {
            torch::Tensor small_test = torch::hann_window(i, periodic);
        }

        if (offset < Size) {
            int64_t random_length = std::abs(static_cast<int64_t>(Data[offset])) % 1000 + 1;
            torch::Tensor random_result = torch::hann_window(random_length, periodic);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}