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
        
        if (offset >= Size) {
            return 0;
        }
        
        if (input_tensor.dim() < 2) {
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0);
        }
        
        uint8_t q_byte = Data[offset % Size];
        offset++;
        int64_t q = static_cast<int64_t>(q_byte % 20);
        
        uint8_t niter_byte = Data[offset % Size];
        offset++;
        int64_t niter = static_cast<int64_t>(niter_byte % 10) + 1;
        
        uint8_t M_byte = Data[offset % Size];
        offset++;
        int64_t M = static_cast<int64_t>(M_byte % 10);
        
        auto result = torch::svd_lowrank(input_tensor, q, niter, M);
        
        auto U = std::get<0>(result);
        auto S = std::get<1>(result);
        auto V = std::get<2>(result);
        
        if (U.numel() > 0) {
            auto sum_U = torch::sum(U);
        }
        if (S.numel() > 0) {
            auto sum_S = torch::sum(S);
        }
        if (V.numel() > 0) {
            auto sum_V = torch::sum(V);
        }
        
        if (offset < Size) {
            uint8_t variant_selector = Data[offset % Size];
            offset++;
            
            if (variant_selector % 4 == 0) {
                auto result2 = torch::svd_lowrank(input_tensor);
            } else if (variant_selector % 4 == 1) {
                auto result3 = torch::svd_lowrank(input_tensor, q);
            } else if (variant_selector % 4 == 2) {
                auto result4 = torch::svd_lowrank(input_tensor, q, niter);
            } else {
                int64_t negative_q = -static_cast<int64_t>(q_byte);
                auto result5 = torch::svd_lowrank(input_tensor, negative_q, niter, M);
            }
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_result = torch::svd_lowrank(input_tensor, 0, 1, 0);
        }
        
        if (input_tensor.dim() >= 2) {
            int64_t min_dim = std::min(input_tensor.size(-2), input_tensor.size(-1));
            int64_t large_q = min_dim + 10;
            auto large_q_result = torch::svd_lowrank(input_tensor, large_q, 1, 0);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::svd_lowrank(zero_tensor, q, niter, M);
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
            auto inf_result = torch::svd_lowrank(inf_tensor, std::min(q, static_cast<int64_t>(5)), 1, 0);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
            auto nan_result = torch::svd_lowrank(nan_tensor, std::min(q, static_cast<int64_t>(5)), 1, 0);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}