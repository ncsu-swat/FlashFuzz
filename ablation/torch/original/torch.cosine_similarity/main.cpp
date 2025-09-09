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

        auto x1 = fuzzer_utils::createTensor(Data, Size, offset);
        auto x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        int64_t dim = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        double eps = 1e-8;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (eps <= 0.0 || !std::isfinite(eps)) {
                eps = 1e-8;
            }
        }
        
        torch::Tensor result = torch::cosine_similarity(x1, x2, dim, eps);
        
        if (x1.numel() == 0 || x2.numel() == 0) {
            torch::Tensor empty_result = torch::cosine_similarity(x1, x2, dim, eps);
        }
        
        if (x1.dim() > 0 && x2.dim() > 0) {
            int64_t max_dim = std::max(x1.dim(), x2.dim()) - 1;
            int64_t min_dim = -std::max(x1.dim(), x2.dim());
            int64_t clamped_dim = std::max(min_dim, std::min(max_dim, dim));
            torch::Tensor clamped_result = torch::cosine_similarity(x1, x2, clamped_dim, eps);
        }
        
        torch::Tensor extreme_eps_result = torch::cosine_similarity(x1, x2, dim, 1e-20);
        torch::Tensor large_eps_result = torch::cosine_similarity(x1, x2, dim, 1e20);
        
        if (x1.dtype() != x2.dtype()) {
            auto promoted_x1 = x1.to(torch::kFloat);
            auto promoted_x2 = x2.to(torch::kFloat);
            torch::Tensor promoted_result = torch::cosine_similarity(promoted_x1, promoted_x2, dim, eps);
        }
        
        auto x1_inf = x1.clone();
        auto x2_inf = x2.clone();
        if (x1_inf.numel() > 0) {
            x1_inf.flatten()[0] = std::numeric_limits<float>::infinity();
        }
        if (x2_inf.numel() > 0) {
            x2_inf.flatten()[0] = -std::numeric_limits<float>::infinity();
        }
        torch::Tensor inf_result = torch::cosine_similarity(x1_inf, x2_inf, dim, eps);
        
        auto x1_nan = x1.clone();
        auto x2_nan = x2.clone();
        if (x1_nan.numel() > 0) {
            x1_nan.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
        }
        if (x2_nan.numel() > 0) {
            x2_nan.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
        }
        torch::Tensor nan_result = torch::cosine_similarity(x1_nan, x2_nan, dim, eps);
        
        if (x1.dim() > 0 && x2.dim() > 0) {
            for (int64_t test_dim = -x1.dim(); test_dim < x1.dim(); ++test_dim) {
                torch::Tensor dim_result = torch::cosine_similarity(x1, x2, test_dim, eps);
            }
        }
        
        auto zero_x1 = torch::zeros_like(x1);
        auto zero_x2 = torch::zeros_like(x2);
        torch::Tensor zero_result = torch::cosine_similarity(zero_x1, zero_x2, dim, eps);
        
        try {
            auto x1_broadcasted = x1.expand({-1, -1});
            auto x2_broadcasted = x2.expand({-1, -1});
            torch::Tensor broadcast_result = torch::cosine_similarity(x1_broadcasted, x2_broadcasted, dim, eps);
        } catch (...) {
        }
        
        if (x1.dim() == x2.dim() && x1.dim() > 0) {
            auto sizes1 = x1.sizes().vec();
            auto sizes2 = x2.sizes().vec();
            for (size_t i = 0; i < sizes1.size(); ++i) {
                if (sizes1[i] != sizes2[i] && sizes1[i] != 1 && sizes2[i] != 1) {
                    break;
                }
            }
            torch::Tensor broadcast_test = torch::cosine_similarity(x1, x2, dim, eps);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}