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
        
        auto result = torch::isposinf(input_tensor);
        
        if (offset < Size) {
            uint8_t out_selector = Data[offset++];
            torch::Tensor out_tensor;
            
            if (out_selector % 2 == 0) {
                out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::isposinf_out(out_tensor, input_tensor);
            }
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_result = torch::isposinf(input_tensor.flatten()[0]);
        }
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            
            auto inf_result = torch::isposinf(inf_tensor);
            auto neg_inf_result = torch::isposinf(neg_inf_tensor);
            auto nan_result = torch::isposinf(nan_tensor);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_inf = torch::full_like(input_tensor, std::complex<double>(std::numeric_limits<double>::infinity(), 0.0));
            auto complex_result = torch::isposinf(complex_inf);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::isposinf(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        auto ones_result = torch::isposinf(ones_tensor);
        
        if (input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::isposinf(reshaped);
        }
        
        if (input_tensor.numel() > 1) {
            auto sliced = input_tensor.flatten().slice(0, 0, 1);
            auto sliced_result = torch::isposinf(sliced);
        }
        
        auto detached = input_tensor.detach();
        auto detached_result = torch::isposinf(detached);
        
        if (input_tensor.is_floating_point()) {
            auto clamped = torch::clamp(input_tensor, -1e10, 1e10);
            auto clamped_result = torch::isposinf(clamped);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}