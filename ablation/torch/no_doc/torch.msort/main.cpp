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
        
        uint8_t config_byte = Data[offset++];
        
        bool stable = (config_byte & 0x01) != 0;
        bool use_dim = (config_byte & 0x02) != 0;
        bool use_descending = (config_byte & 0x04) != 0;
        
        if (use_dim && input_tensor.dim() > 0) {
            if (offset >= Size) {
                return 0;
            }
            
            uint8_t dim_byte = Data[offset++];
            int64_t dim = static_cast<int64_t>(dim_byte) % (2 * input_tensor.dim()) - input_tensor.dim();
            
            if (use_descending) {
                torch::msort(input_tensor, dim, !stable);
            } else {
                torch::msort(input_tensor, dim, stable);
            }
        } else {
            if (use_descending) {
                torch::msort(input_tensor, -1, !stable);
            } else {
                torch::msort(input_tensor, -1, stable);
            }
        }
        
        auto result1 = torch::msort(input_tensor);
        
        if (input_tensor.dim() > 0) {
            auto result2 = torch::msort(input_tensor, 0);
            auto result3 = torch::msort(input_tensor, -1);
            
            if (input_tensor.dim() > 1) {
                auto result4 = torch::msort(input_tensor, 1);
                auto result5 = torch::msort(input_tensor, input_tensor.dim() - 1);
            }
        }
        
        auto result_stable = torch::msort(input_tensor, -1, true);
        auto result_unstable = torch::msort(input_tensor, -1, false);
        
        if (input_tensor.numel() == 0) {
            auto empty_result = torch::msort(input_tensor);
        }
        
        if (input_tensor.numel() == 1) {
            auto single_result = torch::msort(input_tensor);
        }
        
        auto cloned_tensor = input_tensor.clone();
        auto cloned_result = torch::msort(cloned_tensor);
        
        if (input_tensor.is_contiguous()) {
            auto non_contiguous = input_tensor.transpose(0, std::min(1L, input_tensor.dim() - 1));
            if (!non_contiguous.is_contiguous() && non_contiguous.dim() > 0) {
                auto nc_result = torch::msort(non_contiguous);
            }
        }
        
        if (input_tensor.dim() >= 2) {
            for (int64_t d = 0; d < input_tensor.dim(); ++d) {
                auto dim_result = torch::msort(input_tensor, d);
            }
        }
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto float_tensor = input_tensor.to(torch::kFloat);
            auto float_result = torch::msort(float_tensor);
        }
        
        if (input_tensor.dtype() == torch::kInt32 || input_tensor.dtype() == torch::kInt64) {
            auto int_result = torch::msort(input_tensor);
        }
        
        auto reshaped = input_tensor.view({-1});
        auto reshaped_result = torch::msort(reshaped);
        
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            auto sliced = input_tensor.slice(0, 0, std::min(2L, input_tensor.size(0)));
            auto sliced_result = torch::msort(sliced);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}