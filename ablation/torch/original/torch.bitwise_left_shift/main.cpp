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

        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        if (!input_tensor.dtype().isIntegral()) {
            auto integral_types = {torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64};
            auto target_type = *(integral_types.begin() + (Data[offset] % integral_types.size()));
            input_tensor = input_tensor.to(target_type);
            offset++;
        }
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t shift_mode = Data[offset++] % 4;
        
        torch::Tensor result;
        
        if (shift_mode == 0) {
            torch::Tensor other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (!other_tensor.dtype().isIntegral()) {
                auto integral_types = {torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64};
                auto target_type = *(integral_types.begin() + (Data[offset % Size] % integral_types.size()));
                other_tensor = other_tensor.to(target_type);
            }
            result = torch::bitwise_left_shift(input_tensor, other_tensor);
        } else if (shift_mode == 1) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t shift_scalar;
                std::memcpy(&shift_scalar, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                shift_scalar = shift_scalar % 65;
                if (shift_scalar < 0) shift_scalar = -shift_scalar;
                result = torch::bitwise_left_shift(input_tensor, shift_scalar);
            } else {
                result = torch::bitwise_left_shift(input_tensor, 1);
            }
        } else if (shift_mode == 2) {
            torch::Tensor other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (!other_tensor.dtype().isIntegral()) {
                auto integral_types = {torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64};
                auto target_type = *(integral_types.begin() + (Data[offset % Size] % integral_types.size()));
                other_tensor = other_tensor.to(target_type);
            }
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            result = torch::bitwise_left_shift_out(out_tensor, input_tensor, other_tensor);
        } else {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t shift_scalar;
                std::memcpy(&shift_scalar, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                shift_scalar = shift_scalar % 65;
                if (shift_scalar < 0) shift_scalar = -shift_scalar;
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                result = torch::bitwise_left_shift_out(out_tensor, input_tensor, shift_scalar);
            } else {
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                result = torch::bitwise_left_shift_out(out_tensor, input_tensor, 1);
            }
        }
        
        if (result.numel() > 0) {
            auto sum = torch::sum(result);
            volatile auto dummy = sum.item<double>();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}