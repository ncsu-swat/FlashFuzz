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
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        auto result = torch::cos(input_tensor);
        
        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.numel() == result.numel()) {
                torch::cos_out(out_tensor, input_tensor);
            }
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::cos(input_tensor);
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_input = input_tensor.flatten()[0];
            auto scalar_result = torch::cos(scalar_input);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::cos(zero_tensor);
        
        auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
        if (input_tensor.dtype().isFloatingType()) {
            auto inf_result = torch::cos(inf_tensor);
        }
        
        auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
        if (input_tensor.dtype().isFloatingType()) {
            auto neg_inf_result = torch::cos(neg_inf_tensor);
        }
        
        auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
        if (input_tensor.dtype().isFloatingType()) {
            auto nan_result = torch::cos(nan_tensor);
        }
        
        auto large_tensor = torch::full_like(input_tensor, 1e10);
        if (input_tensor.dtype().isFloatingType()) {
            auto large_result = torch::cos(large_tensor);
        }
        
        auto small_tensor = torch::full_like(input_tensor, 1e-10);
        if (input_tensor.dtype().isFloatingType()) {
            auto small_result = torch::cos(small_tensor);
        }
        
        if (input_tensor.requires_grad() == false && input_tensor.dtype().isFloatingType()) {
            auto grad_tensor = input_tensor.clone().detach().requires_grad_(true);
            auto grad_result = torch::cos(grad_tensor);
            if (grad_result.numel() > 0) {
                auto sum_result = grad_result.sum();
                sum_result.backward();
            }
        }
        
        if (input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::cos(reshaped);
        }
        
        if (input_tensor.is_contiguous() == false) {
            auto non_contiguous = input_tensor.transpose(0, input_tensor.dim() > 1 ? 1 : 0);
            auto non_contiguous_result = torch::cos(non_contiguous);
        }
        
        auto pi_tensor = torch::full_like(input_tensor, M_PI);
        if (input_tensor.dtype().isFloatingType()) {
            auto pi_result = torch::cos(pi_tensor);
        }
        
        auto half_pi_tensor = torch::full_like(input_tensor, M_PI / 2.0);
        if (input_tensor.dtype().isFloatingType()) {
            auto half_pi_result = torch::cos(half_pi_tensor);
        }
        
        auto two_pi_tensor = torch::full_like(input_tensor, 2.0 * M_PI);
        if (input_tensor.dtype().isFloatingType()) {
            auto two_pi_result = torch::cos(two_pi_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}