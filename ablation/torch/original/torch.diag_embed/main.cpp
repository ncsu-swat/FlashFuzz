#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.unsqueeze(0);
        }

        if (offset >= Size) {
            return 0;
        }

        int32_t offset_param = 0;
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&offset_param, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }

        int64_t dim1 = -2;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }

        int64_t dim2 = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }

        auto result1 = torch::diag_embed(input_tensor);

        auto result2 = torch::diag_embed(input_tensor, offset_param);

        auto result3 = torch::diag_embed(input_tensor, offset_param, dim1);

        auto result4 = torch::diag_embed(input_tensor, offset_param, dim1, dim2);

        auto result5 = torch::diag_embed(input_tensor, 0, -2, -1);

        auto result6 = torch::diag_embed(input_tensor, 1, 0, 2);

        auto result7 = torch::diag_embed(input_tensor, -1, 1, 0);

        if (input_tensor.dim() >= 2) {
            auto result8 = torch::diag_embed(input_tensor, 0, 0, 1);
            auto result9 = torch::diag_embed(input_tensor, 2, -1, -2);
        }

        auto large_offset_pos = torch::diag_embed(input_tensor, 100);
        auto large_offset_neg = torch::diag_embed(input_tensor, -100);

        if (input_tensor.dim() >= 3) {
            auto result_high_dims = torch::diag_embed(input_tensor, 0, 1, 3);
        }

        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::diag_embed(zero_tensor);

        auto ones_tensor = torch::ones_like(input_tensor);
        auto ones_result = torch::diag_embed(ones_tensor);

        if (input_tensor.numel() > 0) {
            auto single_elem = input_tensor.flatten().slice(0, 0, 1);
            auto single_result = torch::diag_embed(single_elem);
        }

        auto squeezed = input_tensor.squeeze();
        if (squeezed.dim() > 0) {
            auto squeezed_result = torch::diag_embed(squeezed);
        }

        if (input_tensor.dim() == 1 && input_tensor.size(0) > 1) {
            auto partial = input_tensor.slice(0, 0, input_tensor.size(0) / 2);
            auto partial_result = torch::diag_embed(partial);
        }

        auto contiguous_tensor = input_tensor.contiguous();
        auto contiguous_result = torch::diag_embed(contiguous_tensor);

        if (input_tensor.dim() >= 2) {
            auto transposed = input_tensor.transpose(-1, -2);
            auto transposed_result = torch::diag_embed(transposed);
        }

        auto cloned_tensor = input_tensor.clone();
        auto cloned_result = torch::diag_embed(cloned_tensor);

        if (input_tensor.is_floating_point()) {
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
            auto nan_result = torch::diag_embed(nan_tensor);
            
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
            auto inf_result = torch::diag_embed(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<float>::infinity());
            auto neg_inf_result = torch::diag_embed(neg_inf_tensor);
        }

        if (input_tensor.is_complex()) {
            auto complex_result = torch::diag_embed(input_tensor, 1, -2, -1);
        }

        auto detached_tensor = input_tensor.detach();
        auto detached_result = torch::diag_embed(detached_tensor);

        if (input_tensor.dim() >= 4) {
            auto high_dim_result1 = torch::diag_embed(input_tensor, 0, 0, 3);
            auto high_dim_result2 = torch::diag_embed(input_tensor, 0, 1, 2);
        }

        auto extreme_dims_result1 = torch::diag_embed(input_tensor, offset_param, -input_tensor.dim(), -1);
        auto extreme_dims_result2 = torch::diag_embed(input_tensor, offset_param, 0, input_tensor.dim());

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}