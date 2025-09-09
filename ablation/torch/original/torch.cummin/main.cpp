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
        
        uint8_t dim_byte = Data[offset++];
        int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_byte));
        
        if (input_tensor.dim() == 0) {
            dim = 0;
        } else {
            dim = dim % input_tensor.dim();
            if (dim < 0) {
                dim += input_tensor.dim();
            }
        }
        
        auto result = torch::cummin(input_tensor, dim);
        auto values = std::get<0>(result);
        auto indices = std::get<1>(result);
        
        if (values.numel() > 0) {
            auto sum_values = torch::sum(values);
            auto sum_indices = torch::sum(indices);
        }
        
        if (input_tensor.dim() > 0 && input_tensor.size(dim) > 1) {
            auto first_slice = input_tensor.select(dim, 0);
            auto last_values_slice = values.select(dim, input_tensor.size(dim) - 1);
        }
        
        if (offset < Size) {
            uint8_t test_byte = Data[offset++];
            if (test_byte % 4 == 0 && input_tensor.dim() > 0) {
                int64_t neg_dim = -1 - (test_byte % input_tensor.dim());
                auto result_neg = torch::cummin(input_tensor, neg_dim);
            }
            
            if (test_byte % 4 == 1) {
                auto empty_tensor = torch::empty({0}, input_tensor.options());
                if (empty_tensor.dim() > 0) {
                    auto empty_result = torch::cummin(empty_tensor, 0);
                }
            }
            
            if (test_byte % 4 == 2 && input_tensor.dim() > 1) {
                for (int64_t d = 0; d < input_tensor.dim(); ++d) {
                    auto result_d = torch::cummin(input_tensor, d);
                }
            }
            
            if (test_byte % 4 == 3) {
                auto large_tensor = torch::ones({1000}, input_tensor.options());
                auto large_result = torch::cummin(large_tensor, 0);
            }
        }
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto inf_result = torch::cummin(inf_tensor, dim);
            
            auto ninf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto ninf_result = torch::cummin(ninf_tensor, dim);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto nan_result = torch::cummin(nan_tensor, dim);
        }
        
        if (input_tensor.dtype().isIntegralType(false)) {
            auto max_val = torch::full_like(input_tensor, std::numeric_limits<int64_t>::max());
            auto max_result = torch::cummin(max_val, dim);
            
            auto min_val = torch::full_like(input_tensor, std::numeric_limits<int64_t>::min());
            auto min_result = torch::cummin(min_val, dim);
        }
        
        if (input_tensor.dim() > 0) {
            auto single_elem = input_tensor.select(dim, 0).unsqueeze(dim);
            auto single_result = torch::cummin(single_elem, dim);
        }
        
        auto contiguous_tensor = input_tensor.contiguous();
        auto contiguous_result = torch::cummin(contiguous_tensor, dim);
        
        if (input_tensor.dim() > 1) {
            auto transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
            int64_t trans_dim = (dim == 0) ? input_tensor.dim() - 1 : (dim == input_tensor.dim() - 1) ? 0 : dim;
            auto trans_result = torch::cummin(transposed, trans_dim);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}