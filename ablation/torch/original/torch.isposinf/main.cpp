#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        auto result = torch::isposinf(input_tensor);
        
        if (offset < Size) {
            uint8_t out_flag = Data[offset++];
            if (out_flag % 2 == 1) {
                auto out_tensor = torch::empty_like(result);
                torch::isposinf_out(out_tensor, input_tensor);
            }
        }
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::isposinf(input_tensor2);
        }
        
        auto scalar_tensor = torch::tensor(std::numeric_limits<double>::infinity());
        auto scalar_result = torch::isposinf(scalar_tensor);
        
        auto neg_inf_tensor = torch::tensor(-std::numeric_limits<double>::infinity());
        auto neg_inf_result = torch::isposinf(neg_inf_tensor);
        
        auto nan_tensor = torch::tensor(std::numeric_limits<double>::quiet_NaN());
        auto nan_result = torch::isposinf(nan_tensor);
        
        auto zero_tensor = torch::zeros({0});
        auto zero_result = torch::isposinf(zero_tensor);
        
        if (input_tensor.numel() > 0) {
            auto first_elem = input_tensor.flatten()[0];
            auto single_elem_tensor = torch::tensor(first_elem);
            auto single_result = torch::isposinf(single_elem_tensor);
        }
        
        auto large_tensor = torch::full({1}, 1e308);
        auto large_result = torch::isposinf(large_tensor);
        
        auto small_tensor = torch::full({1}, 1e-308);
        auto small_result = torch::isposinf(small_tensor);
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto modified_tensor = input_tensor.clone();
            if (modified_tensor.numel() > 0) {
                modified_tensor.flatten()[0] = std::numeric_limits<double>::infinity();
                auto modified_result = torch::isposinf(modified_tensor);
            }
        }
        
        if (input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::isposinf(reshaped);
        }
        
        auto complex_tensor = torch::complex(input_tensor.to(torch::kFloat), torch::zeros_like(input_tensor.to(torch::kFloat)));
        auto complex_result = torch::isposinf(complex_tensor);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}