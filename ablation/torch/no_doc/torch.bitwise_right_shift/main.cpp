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
        
        if (!tensor1.dtype().is_integral() || !tensor2.dtype().is_integral()) {
            return 0;
        }
        
        torch::bitwise_right_shift(tensor1, tensor2);
        
        if (offset < Size) {
            uint8_t shift_amount = Data[offset];
            torch::bitwise_right_shift(tensor1, shift_amount);
        }
        
        torch::bitwise_right_shift_out(tensor1, tensor1, tensor2);
        
        tensor1.bitwise_right_shift_(tensor2);
        
        if (offset + 1 < Size) {
            int64_t scalar_shift = static_cast<int64_t>(Data[offset + 1]);
            tensor1.bitwise_right_shift_(scalar_shift);
        }
        
        auto result1 = tensor1 >> tensor2;
        
        if (offset + 2 < Size) {
            int shift_val = static_cast<int>(Data[offset + 2]);
            auto result2 = tensor1 >> shift_val;
        }
        
        if (tensor1.numel() > 0 && tensor2.numel() > 0) {
            try {
                auto broadcasted_result = torch::bitwise_right_shift(tensor1, tensor2);
            } catch (...) {
            }
        }
        
        auto zero_tensor = torch::zeros_like(tensor1);
        torch::bitwise_right_shift(tensor1, zero_tensor);
        
        auto ones_tensor = torch::ones_like(tensor1);
        torch::bitwise_right_shift(tensor1, ones_tensor);
        
        if (tensor1.numel() > 0) {
            auto max_val = torch::max(tensor1);
            if (max_val.item<int64_t>() > 0) {
                torch::bitwise_right_shift(tensor1, max_val);
            }
        }
        
        auto negative_tensor = -torch::abs(tensor1);
        try {
            torch::bitwise_right_shift(negative_tensor, tensor2);
        } catch (...) {
        }
        
        if (offset + 3 < Size) {
            int large_shift = 64 + static_cast<int>(Data[offset + 3]);
            try {
                torch::bitwise_right_shift(tensor1, large_shift);
            } catch (...) {
            }
        }
        
        if (offset + 4 < Size) {
            int negative_shift = -static_cast<int>(Data[offset + 4]);
            try {
                torch::bitwise_right_shift(tensor1, negative_shift);
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