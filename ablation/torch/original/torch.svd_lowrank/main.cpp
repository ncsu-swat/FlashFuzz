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

        auto A = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (A.dim() < 2) {
            A = A.unsqueeze(0).unsqueeze(0);
        }
        
        if (A.dim() > 2) {
            auto sizes = A.sizes().vec();
            while (sizes.size() > 2) {
                sizes.pop_back();
            }
            A = A.view(sizes);
        }
        
        if (A.size(0) == 0 || A.size(1) == 0) {
            A = torch::randn({2, 3}, A.options());
        }
        
        int64_t min_dim = std::min(A.size(0), A.size(1));
        
        if (offset < Size) {
            uint8_t q_byte = Data[offset++];
            int q = static_cast<int>(q_byte % std::max(1L, min_dim + 5));
            
            uint8_t niter_byte = offset < Size ? Data[offset++] : 2;
            int niter = static_cast<int>(niter_byte % 10);
            
            auto result = torch::svd_lowrank(A, q, niter);
            auto U = std::get<0>(result);
            auto S = std::get<1>(result);
            auto V = std::get<2>(result);
            
            if (offset < Size) {
                bool use_M = Data[offset++] % 2 == 0;
                if (use_M && offset < Size) {
                    auto M = fuzzer_utils::createTensor(Data, Size, offset);
                    if (M.dim() < 2) {
                        M = M.unsqueeze(0).unsqueeze(0);
                    }
                    if (M.dim() > 2) {
                        auto m_sizes = M.sizes().vec();
                        while (m_sizes.size() > 2) {
                            m_sizes.pop_back();
                        }
                        M = M.view(m_sizes);
                    }
                    
                    if (M.size(0) != 1 || M.size(1) != A.size(1)) {
                        M = torch::randn({1, A.size(1)}, A.options());
                    }
                    
                    auto result_with_M = torch::svd_lowrank(A, q, niter, M);
                    auto U_M = std::get<0>(result_with_M);
                    auto S_M = std::get<1>(result_with_M);
                    auto V_M = std::get<2>(result_with_M);
                }
            }
        } else {
            auto result_default = torch::svd_lowrank(A);
            auto U_def = std::get<0>(result_default);
            auto S_def = std::get<1>(result_default);
            auto V_def = std::get<2>(result_default);
        }
        
        if (offset < Size) {
            auto A_batch = A.unsqueeze(0).repeat({2, 1, 1});
            auto result_batch = torch::svd_lowrank(A_batch);
            auto U_batch = std::get<0>(result_batch);
            auto S_batch = std::get<1>(result_batch);
            auto V_batch = std::get<2>(result_batch);
        }
        
        if (offset < Size) {
            auto A_large = torch::randn({100, 50}, A.options());
            uint8_t q_large_byte = Data[offset++];
            int q_large = static_cast<int>(q_large_byte % 30) + 1;
            auto result_large = torch::svd_lowrank(A_large, q_large);
            auto U_large = std::get<0>(result_large);
            auto S_large = std::get<1>(result_large);
            auto V_large = std::get<2>(result_large);
        }
        
        if (offset < Size && A.dtype() == torch::kFloat) {
            auto A_zero = torch::zeros_like(A);
            auto result_zero = torch::svd_lowrank(A_zero);
            auto U_zero = std::get<0>(result_zero);
            auto S_zero = std::get<1>(result_zero);
            auto V_zero = std::get<2>(result_zero);
        }
        
        if (offset < Size) {
            auto A_singular = torch::ones({3, 3}, A.options());
            A_singular[0] = torch::zeros({3}, A.options());
            A_singular[1] = torch::zeros({3}, A.options());
            auto result_singular = torch::svd_lowrank(A_singular);
            auto U_singular = std::get<0>(result_singular);
            auto S_singular = std::get<1>(result_singular);
            auto V_singular = std::get<2>(result_singular);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}