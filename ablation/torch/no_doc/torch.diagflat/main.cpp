#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset < Size) {
            uint8_t offset_byte = Data[offset++];
            int64_t offset_val = static_cast<int64_t>(static_cast<int8_t>(offset_byte));
            torch::diagflat(input_tensor, offset_val);
        } else {
            torch::diagflat(input_tensor);
        }
        
        torch::diagflat(input_tensor, 0);
        torch::diagflat(input_tensor, 1);
        torch::diagflat(input_tensor, -1);
        
        if (input_tensor.numel() > 0) {
            torch::diagflat(input_tensor, input_tensor.numel());
            torch::diagflat(input_tensor, -input_tensor.numel());
        }
        
        auto flattened = input_tensor.flatten();
        torch::diagflat(flattened);
        
        if (input_tensor.dim() > 0) {
            auto squeezed = input_tensor.squeeze();
            torch::diagflat(squeezed);
        }
        
        if (input_tensor.numel() == 1) {
            torch::diagflat(input_tensor, 100);
            torch::diagflat(input_tensor, -100);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        torch::diagflat(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        torch::diagflat(ones_tensor);
        
        if (input_tensor.dtype() != torch::kBool) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
            torch::diagflat(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<float>::infinity());
            torch::diagflat(neg_inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
            torch::diagflat(nan_tensor);
        }
        
        if (input_tensor.is_complex()) {
            torch::diagflat(input_tensor.real());
            torch::diagflat(input_tensor.imag());
        }
        
        if (input_tensor.numel() > 1) {
            auto reshaped = input_tensor.view({-1});
            torch::diagflat(reshaped);
        }
        
        if (offset < Size) {
            uint8_t large_offset_byte = Data[offset++];
            int64_t large_offset = static_cast<int64_t>(large_offset_byte) * 1000;
            torch::diagflat(input_tensor, large_offset);
            torch::diagflat(input_tensor, -large_offset);
        }
        
        auto empty_tensor = torch::empty({0}, input_tensor.options());
        torch::diagflat(empty_tensor);
        
        auto scalar_tensor = torch::tensor(42.0, input_tensor.options());
        torch::diagflat(scalar_tensor);
        
        if (input_tensor.dim() >= 2) {
            auto transposed = input_tensor.transpose(0, 1);
            torch::diagflat(transposed);
        }
        
        if (input_tensor.numel() > 0 && input_tensor.dtype().isFloatingType()) {
            auto clamped = torch::clamp(input_tensor, -1e6, 1e6);
            torch::diagflat(clamped);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}