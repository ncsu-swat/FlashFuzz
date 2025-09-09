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
            torch::msort(input_tensor);
            return 0;
        }
        
        if (input_tensor.dim() == 0) {
            torch::msort(input_tensor);
            return 0;
        }
        
        torch::msort(input_tensor);
        
        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.sizes() == input_tensor.sizes() && out_tensor.dtype() == input_tensor.dtype()) {
                torch::msort(input_tensor, out_tensor);
            }
        }
        
        auto cloned_input = input_tensor.clone();
        torch::msort(cloned_input);
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            torch::msort(input_tensor);
        }
        
        if (input_tensor.dtype() == torch::kBool) {
            torch::msort(input_tensor);
        }
        
        auto reshaped = input_tensor.view({-1});
        if (reshaped.dim() > 0) {
            torch::msort(reshaped);
        }
        
        if (input_tensor.dim() > 1) {
            auto transposed = input_tensor.transpose(0, 1);
            torch::msort(transposed);
        }
        
        if (input_tensor.is_contiguous()) {
            auto non_contiguous = input_tensor.transpose(-1, -2);
            if (non_contiguous.dim() > 0) {
                torch::msort(non_contiguous);
            }
        }
        
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            auto sliced = input_tensor.slice(0, 0, input_tensor.size(0), 2);
            torch::msort(sliced);
        }
        
        auto detached = input_tensor.detach();
        torch::msort(detached);
        
        if (input_tensor.dtype().isFloatingPoint()) {
            auto with_nan = input_tensor.clone();
            if (with_nan.numel() > 0) {
                with_nan.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                torch::msort(with_nan);
            }
            
            auto with_inf = input_tensor.clone();
            if (with_inf.numel() > 0) {
                with_inf.flatten()[0] = std::numeric_limits<float>::infinity();
                torch::msort(with_inf);
            }
        }
        
        if (input_tensor.dtype().isIntegral() && input_tensor.numel() > 0) {
            auto with_extremes = input_tensor.clone();
            if (input_tensor.dtype() == torch::kInt64) {
                with_extremes.flatten()[0] = std::numeric_limits<int64_t>::max();
                if (with_extremes.numel() > 1) {
                    with_extremes.flatten()[1] = std::numeric_limits<int64_t>::min();
                }
            } else if (input_tensor.dtype() == torch::kInt32) {
                with_extremes.flatten()[0] = std::numeric_limits<int32_t>::max();
                if (with_extremes.numel() > 1) {
                    with_extremes.flatten()[1] = std::numeric_limits<int32_t>::min();
                }
            }
            torch::msort(with_extremes);
        }
        
        auto requires_grad_tensor = input_tensor.clone().requires_grad_(true);
        torch::msort(requires_grad_tensor);
        
        if (input_tensor.dim() > 2) {
            std::vector<int64_t> new_shape = {input_tensor.size(0), -1};
            auto reshaped_2d = input_tensor.reshape(new_shape);
            torch::msort(reshaped_2d);
        }
        
        if (input_tensor.dim() == 1 && input_tensor.numel() > 0) {
            auto expanded = input_tensor.unsqueeze(0).expand({3, input_tensor.size(0)});
            torch::msort(expanded);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        torch::msort(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        torch::msort(ones_tensor);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}