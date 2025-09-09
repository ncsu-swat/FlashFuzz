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

        if (input_tensor.numel() == 0) {
            auto result = torch::erfc(input_tensor);
            return 0;
        }

        if (input_tensor.dtype() == torch::kBool || 
            input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kUInt8 || 
            input_tensor.dtype() == torch::kInt16 || 
            input_tensor.dtype() == torch::kInt32 || 
            input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }

        auto result = torch::erfc(input_tensor);

        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor2.numel() > 0) {
                if (input_tensor2.dtype() == torch::kBool || 
                    input_tensor2.dtype() == torch::kInt8 || 
                    input_tensor2.dtype() == torch::kUInt8 || 
                    input_tensor2.dtype() == torch::kInt16 || 
                    input_tensor2.dtype() == torch::kInt32 || 
                    input_tensor2.dtype() == torch::kInt64) {
                    input_tensor2 = input_tensor2.to(torch::kFloat);
                }
                auto result2 = torch::erfc(input_tensor2);
            }
        }

        if (input_tensor.numel() > 0) {
            auto cloned_tensor = input_tensor.clone();
            torch::erfc_(cloned_tensor);
        }

        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::erfc(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto neg_inf_result = torch::erfc(neg_inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::erfc(nan_tensor);
            
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto zero_result = torch::erfc(zero_tensor);
            
            auto large_tensor = torch::full_like(input_tensor, 1e10);
            auto large_result = torch::erfc(large_tensor);
            
            auto small_tensor = torch::full_like(input_tensor, -1e10);
            auto small_result = torch::erfc(small_tensor);
        }

        if (input_tensor.is_complex() && input_tensor.numel() > 0) {
            auto complex_result = torch::erfc(input_tensor);
        }

        if (input_tensor.numel() > 0 && input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::erfc(reshaped);
        }

        if (input_tensor.numel() > 1) {
            auto sliced = input_tensor.slice(0, 0, 1);
            auto sliced_result = torch::erfc(sliced);
        }

        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            auto double_tensor = input_tensor.to(torch::kDouble);
            auto double_result = torch::erfc(double_tensor);
            
            auto float_tensor = input_tensor.to(torch::kFloat);
            auto float_result = torch::erfc(float_tensor);
            
            if (input_tensor.device().is_cpu()) {
                auto half_tensor = input_tensor.to(torch::kHalf);
                auto half_result = torch::erfc(half_tensor);
                
                auto bfloat16_tensor = input_tensor.to(torch::kBFloat16);
                auto bfloat16_result = torch::erfc(bfloat16_tensor);
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