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

        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t numel_byte = Data[offset++];
        int64_t numel = static_cast<int64_t>(numel_byte) + 1;
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t reduce_range_byte = Data[offset++];
        bool reduce_range = (reduce_range_byte % 2) == 1;
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t qscheme_byte = Data[offset++];
        c10::QScheme qscheme;
        switch (qscheme_byte % 4) {
            case 0:
                qscheme = c10::kPerTensorAffine;
                break;
            case 1:
                qscheme = c10::kPerChannelAffine;
                break;
            case 2:
                qscheme = c10::kPerTensorSymmetric;
                break;
            case 3:
                qscheme = c10::kPerChannelSymmetric;
                break;
        }
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t dtype_byte = Data[offset++];
        torch::ScalarType dtype;
        switch (dtype_byte % 3) {
            case 0:
                dtype = torch::kQInt8;
                break;
            case 1:
                dtype = torch::kQUInt8;
                break;
            case 2:
                dtype = torch::kQInt32;
                break;
        }
        
        c10::optional<int64_t> numel_opt;
        if (numel > 0) {
            numel_opt = numel;
        }
        
        auto result = torch::choose_qparams_optimized(input_tensor, numel_opt, reduce_range, qscheme, dtype);
        
        double scale = std::get<0>(result);
        int64_t zero_point = std::get<1>(result);
        
        if (scale <= 0.0 || std::isnan(scale) || std::isinf(scale)) {
            return 0;
        }
        
        if (dtype == torch::kQUInt8) {
            if (zero_point < 0 || zero_point > 255) {
                return 0;
            }
        } else if (dtype == torch::kQInt8) {
            if (zero_point < -128 || zero_point > 127) {
                return 0;
            }
        }
        
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        torch::Tensor flattened = input_tensor.flatten();
        if (flattened.numel() > 0) {
            auto min_val = torch::min(flattened);
            auto max_val = torch::max(flattened);
        }
        
        torch::Tensor large_tensor = torch::randn({1000, 1000});
        auto large_result = torch::choose_qparams_optimized(large_tensor, c10::nullopt, false, c10::kPerTensorAffine, torch::kQUInt8);
        
        torch::Tensor empty_tensor = torch::empty({0});
        auto empty_result = torch::choose_qparams_optimized(empty_tensor, c10::nullopt, true, c10::kPerTensorAffine, torch::kQInt8);
        
        torch::Tensor inf_tensor = torch::full({10}, std::numeric_limits<float>::infinity());
        auto inf_result = torch::choose_qparams_optimized(inf_tensor, c10::nullopt, false, c10::kPerTensorAffine, torch::kQUInt8);
        
        torch::Tensor nan_tensor = torch::full({5}, std::numeric_limits<float>::quiet_NaN());
        auto nan_result = torch::choose_qparams_optimized(nan_tensor, c10::nullopt, true, c10::kPerTensorAffine, torch::kQInt8);
        
        torch::Tensor zero_tensor = torch::zeros({100});
        auto zero_result = torch::choose_qparams_optimized(zero_tensor, c10::nullopt, false, c10::kPerTensorAffine, torch::kQUInt8);
        
        torch::Tensor negative_tensor = torch::full({50}, -1000.0f);
        auto neg_result = torch::choose_qparams_optimized(negative_tensor, c10::nullopt, true, c10::kPerTensorAffine, torch::kQInt8);
        
        torch::Tensor mixed_tensor = torch::cat({torch::full({10}, -1e6f), torch::full({10}, 1e6f)});
        auto mixed_result = torch::choose_qparams_optimized(mixed_tensor, c10::nullopt, false, c10::kPerTensorAffine, torch::kQUInt8);
        
        if (offset < Size) {
            int64_t extreme_numel = static_cast<int64_t>(Data[offset]) * 1000000;
            c10::optional<int64_t> extreme_numel_opt = extreme_numel;
            auto extreme_result = torch::choose_qparams_optimized(input_tensor, extreme_numel_opt, reduce_range, qscheme, dtype);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}