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
        
        auto result = torch::real(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::real(input_tensor2);
        }
        
        if (input_tensor.is_complex()) {
            auto real_part = torch::real(input_tensor);
            auto imag_part = torch::imag(input_tensor);
            auto reconstructed = torch::complex(real_part, imag_part);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_real = torch::real(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        auto ones_real = torch::real(ones_tensor);
        
        if (input_tensor.numel() > 0) {
            auto sliced = input_tensor.slice(0, 0, std::min(static_cast<int64_t>(1), input_tensor.size(0)));
            auto sliced_real = torch::real(sliced);
        }
        
        if (input_tensor.dim() > 1) {
            auto reshaped = input_tensor.reshape({-1});
            auto reshaped_real = torch::real(reshaped);
        }
        
        auto cloned = input_tensor.clone();
        auto cloned_real = torch::real(cloned);
        
        if (input_tensor.is_cuda()) {
            auto cpu_tensor = input_tensor.cpu();
            auto cpu_real = torch::real(cpu_tensor);
        }
        
        auto detached = input_tensor.detach();
        auto detached_real = torch::real(detached);
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto conj_tensor = torch::conj(input_tensor);
            auto conj_real = torch::real(conj_tensor);
        }
        
        if (input_tensor.numel() > 1) {
            auto transposed = input_tensor.t();
            auto transposed_real = torch::real(transposed);
        }
        
        auto contiguous = input_tensor.contiguous();
        auto contiguous_real = torch::real(contiguous);
        
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            auto indexed = input_tensor[0];
            auto indexed_real = torch::real(indexed);
        }
        
        auto squeezed = input_tensor.squeeze();
        auto squeezed_real = torch::real(squeezed);
        
        auto unsqueezed = input_tensor.unsqueeze(0);
        auto unsqueezed_real = torch::real(unsqueezed);
        
        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            auto nan_tensor = input_tensor.clone();
            if (nan_tensor.numel() > 0) {
                nan_tensor.fill_(std::numeric_limits<float>::quiet_NaN());
                auto nan_real = torch::real(nan_tensor);
            }
        }
        
        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            auto inf_tensor = input_tensor.clone();
            if (inf_tensor.numel() > 0) {
                inf_tensor.fill_(std::numeric_limits<float>::infinity());
                auto inf_real = torch::real(inf_tensor);
            }
        }
        
        auto empty_like = torch::empty_like(input_tensor);
        auto empty_real = torch::real(empty_like);
        
        if (input_tensor.dim() > 0) {
            auto permuted = input_tensor.permute({0});
            auto permuted_real = torch::real(permuted);
        }
        
        auto flattened = input_tensor.flatten();
        auto flattened_real = torch::real(flattened);
        
        if (input_tensor.numel() > 0) {
            auto expanded = input_tensor.expand_as(input_tensor);
            auto expanded_real = torch::real(expanded);
        }
        
        if (input_tensor.dim() >= 2) {
            auto view_tensor = input_tensor.view({-1, input_tensor.size(-1)});
            auto view_real = torch::real(view_tensor);
        }
        
        auto abs_tensor = torch::abs(input_tensor);
        auto abs_real = torch::real(abs_tensor);
        
        if (input_tensor.is_complex()) {
            auto angle_tensor = torch::angle(input_tensor);
            auto angle_real = torch::real(angle_tensor);
        }
        
        if (input_tensor.numel() > 0) {
            auto narrow_tensor = input_tensor.narrow(0, 0, 1);
            auto narrow_real = torch::real(narrow_tensor);
        }
        
        auto neg_tensor = torch::neg(input_tensor);
        auto neg_real = torch::real(neg_tensor);
        
        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            auto sqrt_tensor = torch::sqrt(torch::abs(input_tensor));
            auto sqrt_real = torch::real(sqrt_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}