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
        if (numel > 0 && numel <= 1000000) {
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
            torch::Tensor empty_tensor = torch::empty({0}, input_tensor.options());
            auto empty_result = torch::choose_qparams_optimized(empty_tensor, c10::nullopt, reduce_range, qscheme, dtype);
        }
        
        if (input_tensor.dim() > 0) {
            torch::Tensor reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::choose_qparams_optimized(reshaped, numel_opt, reduce_range, qscheme, dtype);
        }
        
        if (input_tensor.dtype().isFloatingPoint()) {
            torch::Tensor inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
            auto inf_result = torch::choose_qparams_optimized(inf_tensor, numel_opt, reduce_range, qscheme, dtype);
            
            torch::Tensor neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<float>::infinity());
            auto neg_inf_result = torch::choose_qparams_optimized(neg_inf_tensor, numel_opt, reduce_range, qscheme, dtype);
            
            torch::Tensor nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
            auto nan_result = torch::choose_qparams_optimized(nan_tensor, numel_opt, reduce_range, qscheme, dtype);
        }
        
        if (input_tensor.numel() > 1) {
            torch::Tensor constant_tensor = torch::full_like(input_tensor, 42.0);
            auto constant_result = torch::choose_qparams_optimized(constant_tensor, numel_opt, reduce_range, qscheme, dtype);
        }
        
        for (int i = 0; i < 4; ++i) {
            c10::QScheme test_qscheme;
            switch (i) {
                case 0: test_qscheme = c10::kPerTensorAffine; break;
                case 1: test_qscheme = c10::kPerChannelAffine; break;
                case 2: test_qscheme = c10::kPerTensorSymmetric; break;
                case 3: test_qscheme = c10::kPerChannelSymmetric; break;
            }
            auto scheme_result = torch::choose_qparams_optimized(input_tensor, numel_opt, reduce_range, test_qscheme, dtype);
        }
        
        for (int i = 0; i < 3; ++i) {
            torch::ScalarType test_dtype;
            switch (i) {
                case 0: test_dtype = torch::kQInt8; break;
                case 1: test_dtype = torch::kQUInt8; break;
                case 2: test_dtype = torch::kQInt32; break;
            }
            auto dtype_result = torch::choose_qparams_optimized(input_tensor, numel_opt, reduce_range, qscheme, test_dtype);
        }
        
        auto reduce_true_result = torch::choose_qparams_optimized(input_tensor, numel_opt, true, qscheme, dtype);
        auto reduce_false_result = torch::choose_qparams_optimized(input_tensor, numel_opt, false, qscheme, dtype);
        
        std::vector<int64_t> test_numels = {1, 10, 100, 1000, 10000};
        for (int64_t test_numel : test_numels) {
            if (test_numel <= input_tensor.numel()) {
                auto numel_result = torch::choose_qparams_optimized(input_tensor, test_numel, reduce_range, qscheme, dtype);
            }
        }
        
        auto nullopt_result = torch::choose_qparams_optimized(input_tensor, c10::nullopt, reduce_range, qscheme, dtype);
        
        if (input_tensor.dtype().isFloatingPoint() && input_tensor.numel() > 0) {
            torch::Tensor extreme_tensor = input_tensor.clone();
            if (extreme_tensor.numel() >= 2) {
                extreme_tensor.view(-1)[0] = -1e10;
                extreme_tensor.view(-1)[-1] = 1e10;
                auto extreme_result = torch::choose_qparams_optimized(extreme_tensor, numel_opt, reduce_range, qscheme, dtype);
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