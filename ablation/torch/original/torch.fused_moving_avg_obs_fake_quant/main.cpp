#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 20) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto scale = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto zero_point = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto running_min = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto running_max = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset + 16 > Size) {
            return 0;
        }
        
        double averaging_const;
        std::memcpy(&averaging_const, Data + offset, sizeof(double));
        offset += sizeof(double);
        
        int64_t quant_min_raw;
        std::memcpy(&quant_min_raw, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        int64_t quant_min = quant_min_raw % 256;
        
        int64_t quant_max_raw;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&quant_max_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            quant_max_raw = quant_min_raw + 1;
        }
        int64_t quant_max = quant_max_raw % 256;
        if (quant_max <= quant_min) {
            quant_max = quant_min + 1;
        }
        
        uint8_t flags = 0;
        if (offset < Size) {
            flags = Data[offset++];
        }
        
        bool per_row_fake_quant = (flags & 0x01) != 0;
        bool symmetric_quant = (flags & 0x02) != 0;
        
        try {
            auto result = torch::fused_moving_avg_obs_fake_quant(
                input_tensor,
                running_min,
                running_max,
                scale,
                zero_point,
                averaging_const,
                quant_min,
                quant_max,
                per_row_fake_quant,
                symmetric_quant
            );
            
            if (result.numel() > 0) {
                auto sum = torch::sum(result);
                if (torch::isfinite(sum).item<bool>()) {
                    volatile float dummy = sum.item<float>();
                    (void)dummy;
                }
            }
        } catch (const c10::Error& e) {
            return 0;
        }
        
        try {
            auto result2 = torch::fused_moving_avg_obs_fake_quant(
                input_tensor,
                running_min,
                running_max,
                scale,
                zero_point,
                -averaging_const,
                quant_max,
                quant_min,
                !per_row_fake_quant,
                !symmetric_quant
            );
        } catch (const c10::Error& e) {
            return 0;
        }
        
        if (input_tensor.numel() > 0) {
            try {
                auto empty_scale = torch::empty({0});
                auto empty_zp = torch::empty({0});
                auto empty_min = torch::empty({0});
                auto empty_max = torch::empty({0});
                
                auto result3 = torch::fused_moving_avg_obs_fake_quant(
                    input_tensor,
                    empty_min,
                    empty_max,
                    empty_scale,
                    empty_zp,
                    0.0,
                    0,
                    255,
                    false,
                    false
                );
            } catch (const c10::Error& e) {
                return 0;
            }
        }
        
        try {
            auto large_tensor = torch::ones({1000, 1000});
            auto large_scale = torch::ones({1000});
            auto large_zp = torch::zeros({1000});
            auto large_min = torch::full({1000}, -1000.0);
            auto large_max = torch::full({1000}, 1000.0);
            
            auto result4 = torch::fused_moving_avg_obs_fake_quant(
                large_tensor,
                large_min,
                large_max,
                large_scale,
                large_zp,
                1e-10,
                -128,
                127,
                true,
                true
            );
        } catch (const c10::Error& e) {
            return 0;
        } catch (const std::bad_alloc& e) {
            return 0;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}