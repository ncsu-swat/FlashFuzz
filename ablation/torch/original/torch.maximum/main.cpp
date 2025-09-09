#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }

        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble ||
            other.dtype() == torch::kComplexFloat || other.dtype() == torch::kComplexDouble) {
            return 0;
        }
        
        torch::Tensor result = torch::maximum(input, other);
        
        if (offset < Size) {
            uint8_t use_out_tensor = Data[offset] % 2;
            if (use_out_tensor) {
                torch::Tensor out = torch::empty_like(result);
                torch::maximum_out(out, input, other);
            }
        }
        
        if (input.numel() == 0 || other.numel() == 0) {
            torch::maximum(input, other);
        }
        
        if (input.sizes() != other.sizes()) {
            torch::maximum(input, other);
        }
        
        auto input_scalar = torch::tensor(1.0);
        auto other_scalar = torch::tensor(2.0);
        torch::maximum(input_scalar, other_scalar);
        
        if (input.numel() > 0) {
            torch::maximum(input, input_scalar);
            torch::maximum(input_scalar, input);
        }
        
        auto nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN());
        auto regular_tensor = torch::ones({2, 2});
        torch::maximum(nan_tensor, regular_tensor);
        torch::maximum(regular_tensor, nan_tensor);
        
        auto inf_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity());
        auto neg_inf_tensor = torch::full({2, 2}, -std::numeric_limits<float>::infinity());
        torch::maximum(inf_tensor, regular_tensor);
        torch::maximum(neg_inf_tensor, regular_tensor);
        torch::maximum(inf_tensor, neg_inf_tensor);
        
        if (input.dtype() != torch::kBool && other.dtype() != torch::kBool) {
            auto zero_tensor = torch::zeros_like(input);
            torch::maximum(input, zero_tensor);
        }
        
        auto large_tensor = torch::full({2, 2}, 1e38f);
        auto small_tensor = torch::full({2, 2}, -1e38f);
        torch::maximum(large_tensor, small_tensor);
        
        if (input.numel() > 1 && other.numel() > 1) {
            auto input_view = input.view(-1);
            auto other_view = other.view(-1);
            if (input_view.numel() == other_view.numel()) {
                torch::maximum(input_view, other_view);
            }
        }
        
        if (input.dim() > 0 && other.dim() > 0) {
            auto input_t = input.transpose(0, input.dim() - 1);
            torch::maximum(input_t, other);
        }
        
        if (input.is_contiguous() && other.is_contiguous()) {
            auto input_nc = input.as_strided(input.sizes(), input.strides());
            torch::maximum(input_nc, other);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}