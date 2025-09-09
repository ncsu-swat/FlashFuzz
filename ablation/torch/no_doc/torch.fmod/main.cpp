#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (tensor1.numel() == 0 || tensor2.numel() == 0) {
            return 0;
        }
        
        if (tensor1.dtype() != tensor2.dtype()) {
            if (tensor1.dtype() == torch::kBool || tensor2.dtype() == torch::kBool) {
                return 0;
            }
            if (tensor1.is_floating_point() != tensor2.is_floating_point()) {
                return 0;
            }
        }
        
        auto result1 = torch::fmod(tensor1, tensor2);
        
        if (tensor1.sizes() != tensor2.sizes()) {
            try {
                auto broadcasted = torch::broadcast_tensors({tensor1, tensor2});
                auto result2 = torch::fmod(std::get<0>(broadcasted), std::get<1>(broadcasted));
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            uint8_t scalar_byte = Data[offset++];
            double scalar_val = static_cast<double>(scalar_byte) / 255.0 * 20.0 - 10.0;
            
            if (scalar_val != 0.0) {
                auto result3 = torch::fmod(tensor1, scalar_val);
                auto result4 = torch::fmod(scalar_val, tensor1);
            }
        }
        
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 0) {
                try {
                    auto tensor1_copy = tensor1.clone();
                    tensor1_copy.fmod_(tensor2);
                } catch (...) {
                }
            }
        }
        
        auto zero_tensor = torch::zeros_like(tensor1);
        try {
            auto result_zero = torch::fmod(tensor1, zero_tensor);
        } catch (...) {
        }
        
        auto ones_tensor = torch::ones_like(tensor1);
        auto result_ones = torch::fmod(tensor1, ones_tensor);
        
        if (tensor1.is_floating_point()) {
            auto inf_tensor = torch::full_like(tensor1, std::numeric_limits<double>::infinity());
            try {
                auto result_inf1 = torch::fmod(tensor1, inf_tensor);
                auto result_inf2 = torch::fmod(inf_tensor, tensor1);
            } catch (...) {
            }
            
            auto nan_tensor = torch::full_like(tensor1, std::numeric_limits<double>::quiet_NaN());
            try {
                auto result_nan1 = torch::fmod(tensor1, nan_tensor);
                auto result_nan2 = torch::fmod(nan_tensor, tensor1);
            } catch (...) {
            }
        }
        
        if (tensor1.is_signed() && tensor1.dtype() != torch::kBool) {
            auto neg_tensor = -tensor1;
            try {
                auto result_neg1 = torch::fmod(neg_tensor, tensor2);
                auto result_neg2 = torch::fmod(tensor1, -tensor2);
                auto result_neg3 = torch::fmod(neg_tensor, -tensor2);
            } catch (...) {
            }
        }
        
        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
            try {
                auto squeezed1 = tensor1.squeeze();
                auto squeezed2 = tensor2.squeeze();
                auto result_squeezed = torch::fmod(squeezed1, squeezed2);
            } catch (...) {
            }
            
            try {
                auto unsqueezed1 = tensor1.unsqueeze(0);
                auto unsqueezed2 = tensor2.unsqueeze(0);
                auto result_unsqueezed = torch::fmod(unsqueezed1, unsqueezed2);
            } catch (...) {
            }
        }
        
        if (tensor1.numel() > 1) {
            try {
                auto flattened1 = tensor1.flatten();
                auto flattened2 = tensor2.flatten();
                auto result_flat = torch::fmod(flattened1, flattened2);
            } catch (...) {
            }
        }
        
        if (tensor1.is_contiguous() && tensor2.is_contiguous()) {
            try {
                auto transposed1 = tensor1.t();
                auto transposed2 = tensor2.t();
                auto result_transposed = torch::fmod(transposed1, transposed2);
            } catch (...) {
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