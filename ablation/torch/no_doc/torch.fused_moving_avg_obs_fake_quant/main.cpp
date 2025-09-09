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
        int64_t quant_max = (quant_max_raw % 256) + quant_min;
        
        uint8_t flags = 0;
        if (offset < Size) {
            flags = Data[offset];
            offset++;
        }
        
        bool per_row_fake_quant = (flags & 0x01) != 0;
        bool symmetric_quant = (flags & 0x02) != 0;
        
        try {
            input_tensor = input_tensor.to(torch::kFloat);
        } catch (...) {
            return 0;
        }
        
        try {
            scale = scale.to(torch::kFloat);
        } catch (...) {
            return 0;
        }
        
        try {
            zero_point = zero_point.to(torch::kFloat);
        } catch (...) {
            return 0;
        }
        
        try {
            running_min = running_min.to(torch::kFloat);
        } catch (...) {
            return 0;
        }
        
        try {
            running_max = running_max.to(torch::kFloat);
        } catch (...) {
            return 0;
        }
        
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        if (scale.numel() == 0 || zero_point.numel() == 0 || running_min.numel() == 0 || running_max.numel() == 0) {
            return 0;
        }
        
        try {
            if (scale.dim() > 1) {
                scale = scale.flatten();
            }
            if (scale.numel() > 1) {
                scale = scale[0];
            }
            scale = scale.item<float>();
        } catch (...) {
            scale = torch::tensor(1.0, torch::kFloat);
        }
        
        try {
            if (zero_point.dim() > 1) {
                zero_point = zero_point.flatten();
            }
            if (zero_point.numel() > 1) {
                zero_point = zero_point[0];
            }
            zero_point = zero_point.item<float>();
        } catch (...) {
            zero_point = torch::tensor(0.0, torch::kFloat);
        }
        
        try {
            if (running_min.dim() > 1) {
                running_min = running_min.flatten();
            }
            if (running_min.numel() > 1) {
                running_min = running_min[0];
            }
            running_min = running_min.item<float>();
        } catch (...) {
            running_min = torch::tensor(-1.0, torch::kFloat);
        }
        
        try {
            if (running_max.dim() > 1) {
                running_max = running_max.flatten();
            }
            if (running_max.numel() > 1) {
                running_max = running_max[0];
            }
            running_max = running_max.item<float>();
        } catch (...) {
            running_max = torch::tensor(1.0, torch::kFloat);
        }
        
        if (std::isnan(averaging_const) || std::isinf(averaging_const)) {
            averaging_const = 0.01;
        }
        
        if (averaging_const < 0.0) {
            averaging_const = -averaging_const;
        }
        
        if (averaging_const > 1.0) {
            averaging_const = 1.0 / (1.0 + averaging_const);
        }
        
        auto result = torch::fused_moving_avg_obs_fake_quant(
            input_tensor,
            scale,
            zero_point,
            running_min,
            running_max,
            averaging_const,
            quant_min,
            quant_max,
            per_row_fake_quant,
            symmetric_quant
        );
        
        if (result.numel() > 0) {
            auto sum = torch::sum(result);
            if (sum.numel() > 0) {
                auto item = sum.item<float>();
                (void)item;
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}