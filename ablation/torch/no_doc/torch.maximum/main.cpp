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
        
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::maximum(input1, input1);
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::maximum(input1, input2);
        
        if (offset < Size) {
            uint8_t scalar_byte = Data[offset++];
            double scalar_val = static_cast<double>(scalar_byte) - 128.0;
            torch::maximum(input1, scalar_val);
            torch::maximum(scalar_val, input1);
        }
        
        if (offset < Size) {
            torch::Tensor zero_tensor = torch::zeros_like(input1);
            torch::maximum(input1, zero_tensor);
            torch::maximum(zero_tensor, input1);
        }
        
        if (offset < Size) {
            torch::Tensor ones_tensor = torch::ones_like(input1);
            torch::maximum(input1, ones_tensor);
        }
        
        if (offset < Size) {
            torch::Tensor neg_tensor = -torch::abs(input1);
            torch::maximum(input1, neg_tensor);
        }
        
        if (offset < Size) {
            torch::Tensor inf_tensor = torch::full_like(input1, std::numeric_limits<double>::infinity());
            torch::maximum(input1, inf_tensor);
        }
        
        if (offset < Size) {
            torch::Tensor ninf_tensor = torch::full_like(input1, -std::numeric_limits<double>::infinity());
            torch::maximum(input1, ninf_tensor);
        }
        
        if (offset < Size && input1.dtype() == torch::kFloat) {
            torch::Tensor nan_tensor = torch::full_like(input1, std::numeric_limits<double>::quiet_NaN());
            torch::maximum(input1, nan_tensor);
        }
        
        if (offset < Size && input1.numel() > 0) {
            auto reshaped = input1.view({-1});
            torch::maximum(input1, reshaped);
        }
        
        if (offset < Size && input1.dim() > 0) {
            auto squeezed = torch::squeeze(input1);
            torch::maximum(input1, squeezed);
        }
        
        if (offset < Size && input1.numel() > 1) {
            auto transposed = input1.transpose(-1, -2);
            torch::maximum(input1, transposed);
        }
        
        if (offset < Size) {
            auto empty_tensor = torch::empty({0}, input1.options());
            torch::maximum(input1, empty_tensor);
        }
        
        if (offset < Size && input1.numel() > 0) {
            auto single_elem = input1.flatten().slice(0, 0, 1);
            torch::maximum(input1, single_elem);
        }
        
        if (offset < Size) {
            torch::Tensor large_tensor = torch::full_like(input1, 1e10);
            torch::maximum(input1, large_tensor);
        }
        
        if (offset < Size) {
            torch::Tensor small_tensor = torch::full_like(input1, -1e10);
            torch::maximum(input1, small_tensor);
        }
        
        if (offset < Size && input1.dtype().isFloatingPoint()) {
            torch::Tensor eps_tensor = torch::full_like(input1, std::numeric_limits<float>::epsilon());
            torch::maximum(input1, eps_tensor);
        }
        
        if (offset < Size && input1.dtype().isIntegral()) {
            auto max_val = input1.dtype() == torch::kInt64 ? 
                std::numeric_limits<int64_t>::max() : 
                std::numeric_limits<int32_t>::max();
            torch::Tensor max_tensor = torch::full_like(input1, max_val);
            torch::maximum(input1, max_tensor);
        }
        
        if (offset < Size && input1.dtype().isIntegral()) {
            auto min_val = input1.dtype() == torch::kInt64 ? 
                std::numeric_limits<int64_t>::min() : 
                std::numeric_limits<int32_t>::min();
            torch::Tensor min_tensor = torch::full_like(input1, min_val);
            torch::maximum(input1, min_tensor);
        }
        
        if (offset < Size && input1.is_contiguous()) {
            auto non_contiguous = input1.transpose(0, -1);
            torch::maximum(input1, non_contiguous);
        }
        
        if (offset < Size && input1.numel() > 1) {
            std::vector<int64_t> new_shape;
            for (int64_t dim : input1.sizes()) {
                new_shape.push_back(dim);
            }
            new_shape.push_back(1);
            auto unsqueezed = input1.unsqueeze(-1);
            torch::maximum(input1, unsqueezed);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}