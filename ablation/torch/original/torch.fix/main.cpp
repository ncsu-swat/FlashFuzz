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
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.sizes() == result.sizes() && out_tensor.dtype() == result.dtype()) {
                torch::fix_out(out_tensor, input_tensor);
            }
        }

        auto cloned_input = input_tensor.clone();
        cloned_input.fix_();

        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::fix(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto neg_inf_result = torch::fix(neg_inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::fix(nan_tensor);
        }

        if (input_tensor.numel() > 0) {
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto zero_result = torch::fix(zero_tensor);
            
            if (input_tensor.dtype().isFloatingPoint()) {
                auto small_pos = torch::full_like(input_tensor, 0.1);
                auto small_pos_result = torch::fix(small_pos);
                
                auto small_neg = torch::full_like(input_tensor, -0.9);
                auto small_neg_result = torch::fix(small_neg);
                
                auto large_pos = torch::full_like(input_tensor, 1e10);
                auto large_pos_result = torch::fix(large_pos);
                
                auto large_neg = torch::full_like(input_tensor, -1e10);
                auto large_neg_result = torch::fix(large_neg);
            }
        }

        if (input_tensor.dtype().isIntegral()) {
            auto int_result = torch::fix(input_tensor);
        }

        if (input_tensor.dtype() == torch::kBool) {
            auto bool_result = torch::fix(input_tensor);
        }

        if (input_tensor.dtype().isComplex()) {
            auto complex_result = torch::fix(input_tensor);
        }

        if (input_tensor.numel() == 0) {
            auto empty_result = torch::fix(input_tensor);
        }

        if (input_tensor.dim() == 0) {
            auto scalar_result = torch::fix(input_tensor);
        }

        auto contiguous_input = input_tensor.contiguous();
        auto contiguous_result = torch::fix(contiguous_input);
        
        if (input_tensor.dim() > 1) {
            auto transposed = input_tensor.transpose(0, 1);
            auto transposed_result = torch::fix(transposed);
        }

        if (input_tensor.numel() > 1) {
            auto view_input = input_tensor.view(-1);
            auto view_result = torch::fix(view_input);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}