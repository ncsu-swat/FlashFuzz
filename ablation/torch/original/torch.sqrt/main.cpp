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
        
        auto result = torch::sqrt(input_tensor);
        
        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.sizes() == result.sizes() && out_tensor.dtype() == result.dtype()) {
                torch::sqrt_out(out_tensor, input_tensor);
            }
        }
        
        auto input_copy = input_tensor.clone();
        input_copy.sqrt_();
        
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset];
            auto target_dtype = fuzzer_utils::parseDataType(dtype_selector);
            try {
                auto converted_input = input_tensor.to(target_dtype);
                auto converted_result = torch::sqrt(converted_input);
            } catch (...) {
            }
        }
        
        if (input_tensor.numel() > 0) {
            try {
                auto flattened = input_tensor.flatten();
                auto sqrt_flat = torch::sqrt(flattened);
                auto reshaped = sqrt_flat.reshape(input_tensor.sizes());
            } catch (...) {
            }
        }
        
        if (input_tensor.dim() > 0) {
            try {
                auto squeezed = input_tensor.squeeze();
                auto sqrt_squeezed = torch::sqrt(squeezed);
            } catch (...) {
            }
        }
        
        try {
            auto unsqueezed = input_tensor.unsqueeze(0);
            auto sqrt_unsqueezed = torch::sqrt(unsqueezed);
        } catch (...) {
        }
        
        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            try {
                auto nan_tensor = input_tensor.clone();
                if (nan_tensor.numel() > 0) {
                    nan_tensor.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                    auto sqrt_nan = torch::sqrt(nan_tensor);
                }
            } catch (...) {
            }
            
            try {
                auto inf_tensor = input_tensor.clone();
                if (inf_tensor.numel() > 0) {
                    inf_tensor.flatten()[0] = std::numeric_limits<float>::infinity();
                    auto sqrt_inf = torch::sqrt(inf_tensor);
                }
            } catch (...) {
            }
        }
        
        if (input_tensor.is_floating_point()) {
            try {
                auto neg_tensor = input_tensor.clone();
                if (neg_tensor.numel() > 0) {
                    neg_tensor.flatten()[0] = -1.0f;
                    auto sqrt_neg = torch::sqrt(neg_tensor);
                }
            } catch (...) {
            }
        }
        
        try {
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto sqrt_zero = torch::sqrt(zero_tensor);
        } catch (...) {
        }
        
        try {
            auto ones_tensor = torch::ones_like(input_tensor);
            auto sqrt_ones = torch::sqrt(ones_tensor);
        } catch (...) {
        }
        
        if (input_tensor.device().is_cpu() && input_tensor.numel() > 1) {
            try {
                auto sliced = input_tensor.slice(input_tensor.dim() - 1, 0, 1);
                auto sqrt_sliced = torch::sqrt(sliced);
            } catch (...) {
            }
        }
        
        if (input_tensor.is_contiguous()) {
            try {
                auto non_contiguous = input_tensor.transpose(-1, -2);
                if (!non_contiguous.is_contiguous()) {
                    auto sqrt_non_contiguous = torch::sqrt(non_contiguous);
                }
            } catch (...) {
            }
        }
        
        try {
            auto detached = input_tensor.detach();
            auto sqrt_detached = torch::sqrt(detached);
        } catch (...) {
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}