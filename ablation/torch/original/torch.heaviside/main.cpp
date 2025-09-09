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
        
        auto result = torch::heaviside(input_tensor, values_tensor);
        
        if (offset < Size) {
            uint8_t out_flag = Data[offset++];
            if (out_flag % 2 == 1) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::heaviside_out(out_tensor, input_tensor, values_tensor);
            }
        }
        
        if (offset < Size) {
            auto input_scalar = input_tensor.numel() > 0 ? input_tensor.flatten()[0] : torch::tensor(0.0);
            auto values_scalar = values_tensor.numel() > 0 ? values_tensor.flatten()[0] : torch::tensor(1.0);
            auto scalar_result = torch::heaviside(input_scalar, values_scalar);
        }
        
        if (input_tensor.numel() > 0 && values_tensor.numel() > 0) {
            auto broadcasted_result = torch::heaviside(input_tensor, values_tensor);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::heaviside(zero_tensor, values_tensor);
        
        auto pos_tensor = torch::ones_like(input_tensor);
        auto pos_result = torch::heaviside(pos_tensor, values_tensor);
        
        auto neg_tensor = -torch::ones_like(input_tensor);
        auto neg_result = torch::heaviside(neg_tensor, values_tensor);
        
        if (input_tensor.dtype().isFloatingPoint()) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::heaviside(inf_tensor, values_tensor);
            
            auto ninf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto ninf_result = torch::heaviside(ninf_tensor, values_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::heaviside(nan_tensor, values_tensor);
        }
        
        if (input_tensor.numel() > 1) {
            auto single_value = torch::tensor({1.5});
            auto broadcast_result = torch::heaviside(input_tensor, single_value);
        }
        
        if (values_tensor.numel() > 1 && input_tensor.numel() == 1) {
            auto broadcast_result2 = torch::heaviside(input_tensor, values_tensor);
        }
        
        auto empty_input = torch::empty({0});
        auto empty_values = torch::empty({0});
        auto empty_result = torch::heaviside(empty_input, empty_values);
        
        if (input_tensor.dim() > 0) {
            auto reshaped_input = input_tensor.view({-1});
            auto reshaped_result = torch::heaviside(reshaped_input, values_tensor);
        }
        
        if (input_tensor.is_contiguous() && input_tensor.numel() > 0) {
            auto non_contiguous = input_tensor.transpose(-1, -2);
            if (non_contiguous.numel() > 0) {
                auto nc_result = torch::heaviside(non_contiguous, values_tensor);
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