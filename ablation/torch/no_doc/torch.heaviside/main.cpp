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
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto values_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::heaviside(input_tensor, values_tensor);
        
        torch::heaviside_out(input_tensor, input_tensor, values_tensor);
        
        input_tensor.heaviside_(values_tensor);
        
        if (input_tensor.numel() > 0 && values_tensor.numel() > 0) {
            auto broadcasted_input = input_tensor.expand({std::max(input_tensor.size(0), int64_t(1))});
            auto broadcasted_values = values_tensor.expand({std::max(values_tensor.size(0), int64_t(1))});
            torch::heaviside(broadcasted_input, broadcasted_values);
        }
        
        if (input_tensor.dim() > 0) {
            auto reshaped_input = input_tensor.view({-1});
            auto reshaped_values = values_tensor.view({-1});
            torch::heaviside(reshaped_input, reshaped_values);
        }
        
        auto scalar_input = torch::scalar_tensor(0.0);
        auto scalar_values = torch::scalar_tensor(1.0);
        torch::heaviside(scalar_input, scalar_values);
        torch::heaviside(input_tensor, scalar_values);
        torch::heaviside(scalar_input, values_tensor);
        
        if (input_tensor.dtype() != torch::kBool && values_tensor.dtype() != torch::kBool) {
            auto zero_input = torch::zeros_like(input_tensor);
            auto ones_values = torch::ones_like(values_tensor);
            torch::heaviside(zero_input, ones_values);
            
            auto inf_input = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
            auto neg_inf_input = torch::full_like(input_tensor, -std::numeric_limits<float>::infinity());
            torch::heaviside(inf_input, values_tensor);
            torch::heaviside(neg_inf_input, values_tensor);
            
            if (input_tensor.dtype().isFloatingPoint()) {
                auto nan_input = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
                torch::heaviside(nan_input, values_tensor);
            }
        }
        
        if (input_tensor.numel() == 0) {
            torch::heaviside(input_tensor, values_tensor);
        }
        
        if (values_tensor.numel() == 0) {
            torch::heaviside(input_tensor, values_tensor);
        }
        
        auto empty_input = torch::empty({0});
        auto empty_values = torch::empty({0});
        torch::heaviside(empty_input, empty_values);
        
        if (input_tensor.numel() > 1 && values_tensor.numel() == 1) {
            torch::heaviside(input_tensor, values_tensor);
        }
        
        if (input_tensor.numel() == 1 && values_tensor.numel() > 1) {
            torch::heaviside(input_tensor, values_tensor);
        }
        
        auto large_input = torch::full({1000}, 1e10);
        auto large_values = torch::full({1000}, -1e10);
        torch::heaviside(large_input, large_values);
        
        auto tiny_input = torch::full({100}, 1e-10);
        auto tiny_values = torch::full({100}, 1e-10);
        torch::heaviside(tiny_input, tiny_values);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}