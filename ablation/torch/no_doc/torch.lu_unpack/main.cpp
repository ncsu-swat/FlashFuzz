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

        auto LU_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto pivots_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        bool unpack_data = (Data[offset] % 2) == 1;
        offset++;
        
        if (offset >= Size) {
            return 0;
        }
        
        bool unpack_pivots = (Data[offset] % 2) == 1;
        offset++;
        
        if (LU_tensor.dim() < 2) {
            return 0;
        }
        
        if (pivots_tensor.dim() < 1) {
            return 0;
        }
        
        auto LU_sizes = LU_tensor.sizes();
        auto pivots_sizes = pivots_tensor.sizes();
        
        int64_t m = LU_sizes[LU_sizes.size() - 2];
        int64_t n = LU_sizes[LU_sizes.size() - 1];
        int64_t min_mn = std::min(m, n);
        
        if (pivots_sizes[pivots_sizes.size() - 1] != min_mn) {
            return 0;
        }
        
        for (int i = 0; i < LU_sizes.size() - 2; i++) {
            if (i >= pivots_sizes.size() - 1) {
                return 0;
            }
            if (LU_sizes[i] != pivots_sizes[i]) {
                return 0;
            }
        }
        
        if (!LU_tensor.dtype().isFloatingPoint() && !LU_tensor.dtype().isComplexType()) {
            LU_tensor = LU_tensor.to(torch::kFloat);
        }
        
        if (pivots_tensor.dtype() != torch::kInt32 && pivots_tensor.dtype() != torch::kInt64) {
            pivots_tensor = pivots_tensor.to(torch::kInt32);
        }
        
        auto result = torch::lu_unpack(LU_tensor, pivots_tensor, unpack_data, unpack_pivots);
        
        auto P = std::get<0>(result);
        auto L = std::get<1>(result);
        auto U = std::get<2>(result);
        
        if (P.numel() > 0) {
            auto P_sum = torch::sum(P);
        }
        if (L.numel() > 0) {
            auto L_sum = torch::sum(L);
        }
        if (U.numel() > 0) {
            auto U_sum = torch::sum(U);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}