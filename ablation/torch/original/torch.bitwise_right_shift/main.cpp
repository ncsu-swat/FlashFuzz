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

        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        if (!input.dtype().isIntegral()) {
            auto integral_types = {torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64};
            auto target_type = *(integral_types.begin() + (Data[offset] % integral_types.size()));
            input = input.to(target_type);
            offset++;
        }
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t shift_mode = Data[offset++];
        torch::Tensor other;
        
        if (shift_mode % 3 == 0) {
            other = fuzzer_utils::createTensor(Data, Size, offset);
            if (!other.dtype().isIntegral()) {
                auto integral_types = {torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64};
                auto target_type = *(integral_types.begin() + (shift_mode % integral_types.size()));
                other = other.to(target_type);
            }
        } else if (shift_mode % 3 == 1) {
            if (offset + sizeof(int32_t) <= Size) {
                int32_t scalar_shift;
                std::memcpy(&scalar_shift, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                other = torch::tensor(scalar_shift);
            } else {
                other = torch::tensor(1);
            }
        } else {
            if (offset < Size) {
                int8_t shift_val = static_cast<int8_t>(Data[offset++]);
                other = torch::tensor(shift_val);
            } else {
                other = torch::tensor(0);
            }
        }
        
        torch::bitwise_right_shift(input, other);
        
        if (offset < Size) {
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::bitwise_right_shift_out(out_tensor, input, other);
        }
        
        if (offset < Size && input.numel() > 0) {
            auto input_clone = input.clone();
            input_clone.bitwise_right_shift_(other);
        }
        
        if (offset < Size) {
            auto broadcasted_input = input.expand({-1});
            torch::bitwise_right_shift(broadcasted_input, other);
        }
        
        if (offset < Size && other.numel() > 0) {
            auto other_clone = other.clone();
            torch::bitwise_right_shift(input, other_clone);
        }
        
        if (offset < Size) {
            auto negative_shift = torch::tensor(-1);
            torch::bitwise_right_shift(input, negative_shift);
        }
        
        if (offset < Size) {
            auto large_shift = torch::tensor(64);
            torch::bitwise_right_shift(input, large_shift);
        }
        
        if (offset < Size && input.numel() > 0) {
            auto zero_input = torch::zeros_like(input);
            torch::bitwise_right_shift(zero_input, other);
        }
        
        if (offset < Size && input.numel() > 0) {
            auto ones_input = torch::ones_like(input);
            torch::bitwise_right_shift(ones_input, other);
        }
        
        if (offset < Size && input.numel() > 0) {
            auto max_val_input = torch::full_like(input, std::numeric_limits<int32_t>::max());
            torch::bitwise_right_shift(max_val_input, other);
        }
        
        if (offset < Size && input.numel() > 0) {
            auto min_val_input = torch::full_like(input, std::numeric_limits<int32_t>::min());
            torch::bitwise_right_shift(min_val_input, other);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}