#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.dim() != 1) {
            input_tensor = input_tensor.flatten();
        }
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t r_byte = Data[offset++];
        int64_t r = static_cast<int64_t>(r_byte % 10);
        
        if (offset >= Size) {
            return 0;
        }
        
        bool with_replacement = (Data[offset++] % 2) == 1;
        
        torch::combinations(input_tensor, r, with_replacement);
        
        torch::combinations(input_tensor);
        
        torch::combinations(input_tensor, 0, false);
        torch::combinations(input_tensor, 0, true);
        
        if (input_tensor.numel() > 0) {
            torch::combinations(input_tensor, 1, false);
            torch::combinations(input_tensor, 1, true);
        }
        
        int64_t max_r = std::min(static_cast<int64_t>(20), input_tensor.numel() + 5);
        for (int64_t test_r = -2; test_r <= max_r; test_r++) {
            torch::combinations(input_tensor, test_r, false);
            torch::combinations(input_tensor, test_r, true);
        }
        
        auto empty_tensor = torch::empty({0}, input_tensor.options());
        torch::combinations(empty_tensor, r, with_replacement);
        torch::combinations(empty_tensor, 0, false);
        torch::combinations(empty_tensor, 1, true);
        
        auto large_tensor = torch::arange(100, input_tensor.options());
        torch::combinations(large_tensor, r % 5, with_replacement);
        
        if (input_tensor.numel() > 1) {
            int64_t large_r = input_tensor.numel() * 2;
            torch::combinations(input_tensor, large_r, with_replacement);
        }
        
        auto single_element = torch::tensor({42}, input_tensor.options());
        torch::combinations(single_element, r, with_replacement);
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full({3}, std::numeric_limits<float>::infinity(), input_tensor.options());
            torch::combinations(inf_tensor, r % 4, with_replacement);
            
            auto nan_tensor = torch::full({3}, std::numeric_limits<float>::quiet_NaN(), input_tensor.options());
            torch::combinations(nan_tensor, r % 4, with_replacement);
        }
        
        if (input_tensor.dtype().isIntegralType(false)) {
            auto max_int_tensor = torch::full({3}, std::numeric_limits<int64_t>::max(), torch::kInt64);
            torch::combinations(max_int_tensor, r % 4, with_replacement);
            
            auto min_int_tensor = torch::full({3}, std::numeric_limits<int64_t>::min(), torch::kInt64);
            torch::combinations(min_int_tensor, r % 4, with_replacement);
        }
        
        auto duplicate_tensor = torch::full({5}, 1, input_tensor.options());
        torch::combinations(duplicate_tensor, r % 6, with_replacement);
        
        if (input_tensor.numel() >= 2) {
            torch::combinations(input_tensor, input_tensor.numel(), false);
            torch::combinations(input_tensor, input_tensor.numel(), true);
            torch::combinations(input_tensor, input_tensor.numel() + 1, true);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}