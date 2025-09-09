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
        
        auto result = torch::fix(input_tensor);
        
        if (offset < Size) {
            auto inplace_tensor = input_tensor.clone();
            torch::fix_(inplace_tensor);
        }
        
        if (offset < Size) {
            auto out_tensor = torch::empty_like(input_tensor);
            torch::fix_out(out_tensor, input_tensor);
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_result = torch::fix(input_tensor.item());
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::fix(input_tensor);
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_result = torch::fix(input_tensor);
        }
        
        auto large_tensor = torch::randn({1000, 1000}) * 1e10;
        auto large_result = torch::fix(large_tensor);
        
        auto inf_tensor = torch::full({5}, std::numeric_limits<float>::infinity());
        auto inf_result = torch::fix(inf_tensor);
        
        auto nan_tensor = torch::full({5}, std::numeric_limits<float>::quiet_NaN());
        auto nan_result = torch::fix(nan_tensor);
        
        auto negative_tensor = torch::randn({10}) - 5.0;
        auto neg_result = torch::fix(negative_tensor);
        
        auto zero_tensor = torch::zeros({3, 3});
        auto zero_result = torch::fix(zero_tensor);
        
        if (input_tensor.requires_grad()) {
            auto grad_result = torch::fix(input_tensor);
        }
        
        if (input_tensor.device().is_cuda()) {
            auto cuda_result = torch::fix(input_tensor);
        }
        
        auto mixed_tensor = torch::tensor({-3.7, -1.2, 0.0, 1.8, 3.9});
        auto mixed_result = torch::fix(mixed_tensor);
        
        auto very_small = torch::tensor({1e-10, -1e-10});
        auto small_result = torch::fix(very_small);
        
        auto fractional = torch::tensor({0.1, 0.9, -0.1, -0.9});
        auto frac_result = torch::fix(fractional);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}