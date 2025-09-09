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

        auto LU_data = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }
        
        auto LU_pivots = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        uint8_t flags_byte = Data[offset % Size];
        bool unpack_data = (flags_byte & 0x01) != 0;
        bool unpack_pivots = (flags_byte & 0x02) != 0;

        if (LU_data.dim() < 2) {
            auto shape = LU_data.sizes().vec();
            while (shape.size() < 2) {
                shape.push_back(1);
            }
            LU_data = LU_data.reshape(shape);
        }

        int64_t m = LU_data.size(-2);
        int64_t n = LU_data.size(-1);
        int64_t min_mn = std::min(m, n);

        auto batch_dims = LU_data.sizes().slice(0, LU_data.dim() - 2);
        std::vector<int64_t> pivot_shape(batch_dims.begin(), batch_dims.end());
        pivot_shape.push_back(min_mn);
        
        if (LU_pivots.numel() == 0) {
            LU_pivots = torch::ones(pivot_shape, torch::kInt32);
        } else {
            LU_pivots = LU_pivots.to(torch::kInt32);
            if (LU_pivots.dim() == 0) {
                LU_pivots = LU_pivots.unsqueeze(0);
            }
            while (LU_pivots.dim() < pivot_shape.size()) {
                LU_pivots = LU_pivots.unsqueeze(0);
            }
            if (LU_pivots.sizes() != pivot_shape) {
                LU_pivots = LU_pivots.reshape(pivot_shape);
            }
        }

        LU_pivots = torch::clamp(LU_pivots, 1, m);

        auto result = torch::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
        
        auto P = std::get<0>(result);
        auto L = std::get<1>(result);
        auto U = std::get<2>(result);

        if (unpack_data) {
            auto L_sum = L.sum();
            auto U_sum = U.sum();
        }
        
        if (unpack_pivots) {
            auto P_sum = P.sum();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}