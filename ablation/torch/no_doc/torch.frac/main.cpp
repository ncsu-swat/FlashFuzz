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
        
        auto result = torch::frac(input_tensor);
        
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 1) {
                auto input_copy = input_tensor.clone();
                input_copy.frac_();
            }
        }
        
        if (offset < Size) {
            uint8_t out_flag = Data[offset++];
            if (out_flag % 2 == 1 && offset < Size) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::frac_out(out_tensor, input_tensor);
            }
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_result = torch::frac(input_tensor.item());
        }
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::frac(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto neg_inf_result = torch::frac(neg_inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::frac(nan_tensor);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::frac(input_tensor);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::frac(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        auto ones_result = torch::frac(ones_tensor);
        
        if (input_tensor.dtype().isFloatingPoint()) {
            auto large_tensor = torch::full_like(input_tensor, 1e10);
            auto large_result = torch::frac(large_tensor);
            
            auto small_tensor = torch::full_like(input_tensor, 1e-10);
            auto small_result = torch::frac(small_tensor);
            
            auto negative_tensor = torch::full_like(input_tensor, -2.5);
            auto negative_result = torch::frac(negative_tensor);
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_result = torch::frac(input_tensor);
        }
        
        if (input_tensor.dim() == 0) {
            auto scalar_tensor_result = torch::frac(input_tensor);
        }
        
        auto contiguous_tensor = input_tensor.contiguous();
        auto contiguous_result = torch::frac(contiguous_tensor);
        
        if (input_tensor.dim() > 1) {
            auto transposed = input_tensor.transpose(0, 1);
            auto transposed_result = torch::frac(transposed);
        }
        
        if (input_tensor.numel() > 1) {
            auto view_tensor = input_tensor.view(-1);
            auto view_result = torch::frac(view_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}