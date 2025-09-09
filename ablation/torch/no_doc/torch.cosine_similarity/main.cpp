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
        if (offset >= Size) {
            return 0;
        }
        
        auto x2 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t dim_byte = Data[offset % Size];
        offset++;
        
        int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_byte));
        
        uint8_t eps_bytes[4];
        for (int i = 0; i < 4; i++) {
            eps_bytes[i] = Data[(offset + i) % Size];
        }
        offset += 4;
        
        float eps_raw;
        std::memcpy(&eps_raw, eps_bytes, sizeof(float));
        double eps = static_cast<double>(eps_raw);
        if (std::isnan(eps) || std::isinf(eps)) {
            eps = 1e-8;
        }
        if (eps < 0) {
            eps = std::abs(eps);
        }
        if (eps > 1.0) {
            eps = 1e-8;
        }
        
        torch::cosine_similarity(x1, x2);
        
        torch::cosine_similarity(x1, x2, dim);
        
        torch::cosine_similarity(x1, x2, dim, eps);
        
        if (x1.dim() > 0 && x2.dim() > 0) {
            int64_t max_dim = std::max(x1.dim(), x2.dim()) - 1;
            int64_t min_dim = -std::max(x1.dim(), x2.dim());
            
            for (int64_t test_dim = min_dim; test_dim <= max_dim; test_dim++) {
                torch::cosine_similarity(x1, x2, test_dim, eps);
            }
        }
        
        if (x1.numel() > 0 && x2.numel() > 0) {
            auto x1_flat = x1.flatten();
            auto x2_flat = x2.flatten();
            torch::cosine_similarity(x1_flat, x2_flat, 0, eps);
        }
        
        if (x1.dim() >= 2 && x2.dim() >= 2) {
            auto x1_2d = x1.view({-1, x1.size(-1)});
            auto x2_2d = x2.view({-1, x2.size(-1)});
            torch::cosine_similarity(x1_2d, x2_2d, 1, eps);
        }
        
        std::vector<double> eps_values = {0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0};
        for (double test_eps : eps_values) {
            torch::cosine_similarity(x1, x2, dim, test_eps);
        }
        
        if (x1.dtype() != torch::kBool && x2.dtype() != torch::kBool) {
            auto x1_zero = torch::zeros_like(x1);
            auto x2_zero = torch::zeros_like(x2);
            torch::cosine_similarity(x1_zero, x2_zero, dim, eps);
            torch::cosine_similarity(x1, x2_zero, dim, eps);
            torch::cosine_similarity(x1_zero, x2, dim, eps);
        }
        
        if (x1.dtype().isFloatingPoint() && x2.dtype().isFloatingPoint()) {
            auto x1_inf = x1.clone();
            auto x2_inf = x2.clone();
            if (x1_inf.numel() > 0) {
                x1_inf.flatten()[0] = std::numeric_limits<float>::infinity();
            }
            if (x2_inf.numel() > 0) {
                x2_inf.flatten()[0] = -std::numeric_limits<float>::infinity();
            }
            torch::cosine_similarity(x1_inf, x2_inf, dim, eps);
            
            auto x1_nan = x1.clone();
            if (x1_nan.numel() > 0) {
                x1_nan.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
            }
            torch::cosine_similarity(x1_nan, x2, dim, eps);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}