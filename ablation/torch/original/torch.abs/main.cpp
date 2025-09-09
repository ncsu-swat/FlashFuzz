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
        
        auto result = torch::abs(input_tensor);
        
        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.sizes() == result.sizes() && out_tensor.dtype() == result.dtype()) {
                torch::abs_out(out_tensor, input_tensor);
            }
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_input = input_tensor.item();
            auto scalar_result = torch::abs(scalar_input);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto abs_result = torch::abs(input_tensor);
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_result = torch::abs(input_tensor);
        }
        
        auto inplace_tensor = input_tensor.clone();
        inplace_tensor.abs_();
        
        if (input_tensor.requires_grad()) {
            auto grad_input = input_tensor.clone().requires_grad_(true);
            auto grad_result = torch::abs(grad_input);
            if (grad_result.numel() > 0) {
                auto sum_result = grad_result.sum();
                sum_result.backward();
            }
        }
        
        auto detached_input = input_tensor.detach();
        auto detached_result = torch::abs(detached_input);
        
        if (input_tensor.dim() > 0) {
            auto view_input = input_tensor.view(-1);
            auto view_result = torch::abs(view_input);
        }
        
        if (input_tensor.is_contiguous()) {
            auto non_contiguous = input_tensor.transpose(0, input_tensor.dim() > 1 ? 1 : 0);
            auto non_contiguous_result = torch::abs(non_contiguous);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}