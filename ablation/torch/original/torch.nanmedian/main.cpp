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
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t operation_selector = Data[offset++];
        
        if (operation_selector % 2 == 0) {
            auto result = torch::nanmedian(input_tensor);
        } else {
            if (offset >= Size) {
                return 0;
            }
            
            uint8_t dim_selector = Data[offset++];
            int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_selector));
            
            bool keepdim = false;
            if (offset < Size) {
                keepdim = (Data[offset++] % 2) == 1;
            }
            
            auto result = torch::nanmedian(input_tensor, dim, keepdim);
        }
        
        if (input_tensor.numel() > 0 && input_tensor.dim() > 0) {
            for (int64_t d = -input_tensor.dim(); d < input_tensor.dim(); ++d) {
                auto result_dim = torch::nanmedian(input_tensor, d, true);
                auto result_dim_no_keep = torch::nanmedian(input_tensor, d, false);
            }
        }
        
        if (input_tensor.numel() > 0) {
            auto result_scalar = torch::nanmedian(input_tensor);
        }
        
        auto empty_tensor = torch::empty({0}, input_tensor.options());
        if (empty_tensor.numel() == 0) {
            auto empty_result = torch::nanmedian(empty_tensor);
        }
        
        auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
        auto nan_result = torch::nanmedian(nan_tensor);
        
        if (input_tensor.dim() > 0) {
            auto mixed_tensor = input_tensor.clone();
            if (mixed_tensor.numel() > 0 && mixed_tensor.dtype().isFloatingPoint()) {
                mixed_tensor.flatten()[0] = std::numeric_limits<double>::quiet_NaN();
                auto mixed_result = torch::nanmedian(mixed_tensor);
            }
        }
        
        if (input_tensor.dim() > 1) {
            for (int64_t d = 0; d < input_tensor.dim(); ++d) {
                auto result_each_dim = torch::nanmedian(input_tensor, d);
            }
        }
        
        auto single_element = torch::tensor({42.0}, input_tensor.options());
        auto single_result = torch::nanmedian(single_element);
        
        if (input_tensor.dtype().isFloatingPoint()) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::nanmedian(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto neg_inf_result = torch::nanmedian(neg_inf_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}