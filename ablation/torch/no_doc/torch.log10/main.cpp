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
        
        auto result = torch::log10(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::log10(input_tensor2);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::log10(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        auto ones_result = torch::log10(ones_tensor);
        
        auto negative_tensor = -torch::abs(input_tensor);
        auto negative_result = torch::log10(negative_tensor);
        
        auto very_small_tensor = torch::full_like(input_tensor, 1e-10);
        auto very_small_result = torch::log10(very_small_tensor);
        
        auto very_large_tensor = torch::full_like(input_tensor, 1e10);
        auto very_large_result = torch::log10(very_large_tensor);
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::log10(inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::log10(nan_tensor);
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_tensor = input_tensor.flatten()[0];
            auto scalar_result = torch::log10(scalar_tensor);
        }
        
        auto abs_tensor = torch::abs(input_tensor);
        auto abs_result = torch::log10(abs_tensor);
        
        auto exp_tensor = torch::exp(input_tensor);
        auto exp_result = torch::log10(exp_tensor);
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::log10(input_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}