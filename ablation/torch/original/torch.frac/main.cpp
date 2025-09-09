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
            auto result = torch::frac(input_tensor);
            return 0;
        }

        if (input_tensor.dtype() == torch::kBool) {
            auto float_tensor = input_tensor.to(torch::kFloat);
            auto result = torch::frac(float_tensor);
            return 0;
        }

        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto result = torch::frac(input_tensor);
            return 0;
        }

        auto result = torch::frac(input_tensor);

        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.dtype() == input_tensor.dtype() && out_tensor.numel() >= result.numel()) {
                torch::frac_out(out_tensor, input_tensor);
            }
        }

        if (input_tensor.dtype() == torch::kInt8 || input_tensor.dtype() == torch::kUInt8 || 
            input_tensor.dtype() == torch::kInt16 || input_tensor.dtype() == torch::kInt32 || 
            input_tensor.dtype() == torch::kInt64) {
            auto float_input = input_tensor.to(torch::kFloat);
            auto float_result = torch::frac(float_input);
        }

        if (input_tensor.is_cuda()) {
            auto cpu_input = input_tensor.cpu();
            auto cpu_result = torch::frac(cpu_input);
        }

        if (input_tensor.requires_grad()) {
            auto grad_result = torch::frac(input_tensor);
            if (grad_result.numel() > 0) {
                auto grad_output = torch::ones_like(grad_result);
                grad_result.backward(grad_output);
            }
        }

        auto detached_input = input_tensor.detach();
        auto detached_result = torch::frac(detached_input);

        if (input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::frac(reshaped);
        }

        if (input_tensor.numel() > 1) {
            auto sliced = input_tensor.slice(0, 0, 1);
            auto sliced_result = torch::frac(sliced);
        }

        auto cloned_input = input_tensor.clone();
        auto cloned_result = torch::frac(cloned_input);

        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::frac(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto neg_inf_result = torch::frac(neg_inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::frac(nan_tensor);
        }

        if (input_tensor.dtype() == torch::kHalf || input_tensor.dtype() == torch::kBFloat16) {
            auto promoted = input_tensor.to(torch::kFloat);
            auto promoted_result = torch::frac(promoted);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}