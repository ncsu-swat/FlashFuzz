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
        
        auto result = torch::log10(input_tensor);
        
        if (offset < Size) {
            uint8_t out_flag = Data[offset++];
            if (out_flag % 2 == 1) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::log10_out(out_tensor, input_tensor);
            }
        }
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::log10(input_tensor2);
        }
        
        if (offset < Size) {
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto zero_result = torch::log10(zero_tensor);
        }
        
        if (offset < Size) {
            auto negative_tensor = -torch::abs(input_tensor);
            auto neg_result = torch::log10(negative_tensor);
        }
        
        if (offset < Size) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::log10(inf_tensor);
        }
        
        if (offset < Size) {
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::log10(nan_tensor);
        }
        
        if (offset < Size) {
            auto tiny_tensor = torch::full_like(input_tensor, 1e-100);
            auto tiny_result = torch::log10(tiny_tensor);
        }
        
        if (offset < Size) {
            auto huge_tensor = torch::full_like(input_tensor, 1e100);
            auto huge_result = torch::log10(huge_tensor);
        }
        
        if (offset < Size) {
            auto one_tensor = torch::ones_like(input_tensor);
            auto one_result = torch::log10(one_tensor);
        }
        
        if (offset < Size) {
            auto ten_tensor = torch::full_like(input_tensor, 10.0);
            auto ten_result = torch::log10(ten_tensor);
        }
        
        if (offset < Size && input_tensor.numel() > 0) {
            auto scalar_tensor = torch::tensor(input_tensor.item<double>());
            auto scalar_result = torch::log10(scalar_tensor);
        }
        
        if (offset < Size) {
            auto empty_tensor = torch::empty({0}, input_tensor.options());
            auto empty_result = torch::log10(empty_tensor);
        }
        
        if (offset < Size) {
            auto large_shape_tensor = torch::ones({1000, 1000}, input_tensor.options());
            auto large_result = torch::log10(large_shape_tensor);
        }
        
        if (offset < Size) {
            auto complex_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
            auto complex_result = torch::log10(complex_tensor);
        }
        
        if (offset < Size) {
            auto detached_tensor = input_tensor.detach();
            auto detached_result = torch::log10(detached_tensor);
        }
        
        if (offset < Size) {
            auto contiguous_tensor = input_tensor.contiguous();
            auto contiguous_result = torch::log10(contiguous_tensor);
        }
        
        if (offset < Size && input_tensor.dim() > 1) {
            auto transposed_tensor = input_tensor.transpose(0, 1);
            auto transposed_result = torch::log10(transposed_tensor);
        }
        
        if (offset < Size && input_tensor.numel() > 1) {
            auto view_tensor = input_tensor.view(-1);
            auto view_result = torch::log10(view_tensor);
        }
        
        if (offset < Size) {
            auto cloned_tensor = input_tensor.clone();
            auto cloned_result = torch::log10(cloned_tensor);
        }
        
        if (offset < Size) {
            auto requires_grad_tensor = input_tensor.requires_grad_(true);
            auto grad_result = torch::log10(requires_grad_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}