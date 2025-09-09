#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 1) {
            return 0;
        }
        
        int64_t window_length = static_cast<int64_t>(Data[offset++] % 256);
        if (offset >= Size) {
            window_length = 1;
        }
        
        bool periodic = false;
        torch::ScalarType dtype = torch::kFloat;
        torch::Device device = torch::kCPU;
        
        if (offset < Size) {
            periodic = (Data[offset++] % 2) == 1;
        }
        
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        torch::Tensor result1 = torch::hann_window(window_length);
        
        torch::Tensor result2 = torch::hann_window(window_length, periodic);
        
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        torch::Tensor result3 = torch::hann_window(window_length, periodic, options);
        
        torch::Tensor result4 = torch::hann_window(window_length, options);
        
        if (offset < Size) {
            int64_t negative_length = -static_cast<int64_t>(Data[offset++] % 100 + 1);
            torch::Tensor result_neg = torch::hann_window(negative_length);
        }
        
        if (offset < Size) {
            int64_t zero_length = 0;
            torch::Tensor result_zero = torch::hann_window(zero_length);
        }
        
        if (offset < Size) {
            int64_t large_length = static_cast<int64_t>(Data[offset++]) * 1000000;
            torch::Tensor result_large = torch::hann_window(large_length);
        }
        
        std::vector<torch::ScalarType> test_dtypes = {
            torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16,
            torch::kComplexFloat, torch::kComplexDouble
        };
        
        for (auto test_dtype : test_dtypes) {
            auto test_options = torch::TensorOptions().dtype(test_dtype);
            torch::Tensor dtype_result = torch::hann_window(window_length, test_options);
        }
        
        if (offset < Size) {
            uint32_t raw_length;
            if (offset + sizeof(uint32_t) <= Size) {
                std::memcpy(&raw_length, Data + offset, sizeof(uint32_t));
                offset += sizeof(uint32_t);
            } else {
                raw_length = 42;
            }
            int64_t mem_length = static_cast<int64_t>(raw_length);
            torch::Tensor mem_result = torch::hann_window(mem_length);
        }
        
        torch::Tensor extreme_small = torch::hann_window(1);
        torch::Tensor extreme_large = torch::hann_window(65536);
        
        if (offset < Size) {
            int64_t signed_length = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
            torch::Tensor signed_result = torch::hann_window(signed_length);
        }
        
        for (bool per : {true, false}) {
            for (int len = 0; len <= 10; len++) {
                torch::Tensor boundary_result = torch::hann_window(len, per);
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}