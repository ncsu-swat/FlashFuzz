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
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto shift_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble || 
            input_tensor.dtype() == torch::kHalf || input_tensor.dtype() == torch::kBFloat16 ||
            input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            input_tensor = input_tensor.to(torch::kInt64);
        }
        
        if (shift_tensor.dtype() == torch::kFloat || shift_tensor.dtype() == torch::kDouble || 
            shift_tensor.dtype() == torch::kHalf || shift_tensor.dtype() == torch::kBFloat16 ||
            shift_tensor.dtype() == torch::kComplexFloat || shift_tensor.dtype() == torch::kComplexDouble) {
            shift_tensor = shift_tensor.to(torch::kInt64);
        }
        
        torch::bitwise_left_shift(input_tensor, shift_tensor);
        
        if (offset < Size) {
            uint8_t scalar_shift_byte = Data[offset++];
            int64_t scalar_shift = static_cast<int64_t>(scalar_shift_byte) - 128;
            torch::bitwise_left_shift(input_tensor, scalar_shift);
        }
        
        auto input_copy = input_tensor.clone();
        torch::bitwise_left_shift_(input_copy, shift_tensor);
        
        if (input_tensor.numel() > 0 && shift_tensor.numel() > 0) {
            try {
                auto broadcasted_result = torch::bitwise_left_shift(input_tensor, shift_tensor);
            } catch (...) {
            }
        }
        
        if (input_tensor.dim() > 0 && shift_tensor.dim() == 0) {
            torch::bitwise_left_shift(input_tensor, shift_tensor);
        }
        
        if (input_tensor.dim() == 0 && shift_tensor.dim() > 0) {
            torch::bitwise_left_shift(input_tensor, shift_tensor);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        torch::bitwise_left_shift(zero_tensor, shift_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        torch::bitwise_left_shift(ones_tensor, shift_tensor);
        
        if (input_tensor.numel() > 0) {
            auto max_vals = torch::full_like(input_tensor, std::numeric_limits<int64_t>::max());
            try {
                torch::bitwise_left_shift(max_vals, shift_tensor);
            } catch (...) {
            }
            
            auto min_vals = torch::full_like(input_tensor, std::numeric_limits<int64_t>::min());
            try {
                torch::bitwise_left_shift(min_vals, shift_tensor);
            } catch (...) {
            }
        }
        
        if (shift_tensor.numel() > 0) {
            auto large_shifts = torch::full_like(shift_tensor, 64);
            try {
                torch::bitwise_left_shift(input_tensor, large_shifts);
            } catch (...) {
            }
            
            auto negative_shifts = torch::full_like(shift_tensor, -1);
            try {
                torch::bitwise_left_shift(input_tensor, negative_shifts);
            } catch (...) {
            }
        }
        
        if (input_tensor.numel() == 0) {
            torch::bitwise_left_shift(input_tensor, shift_tensor);
        }
        
        if (shift_tensor.numel() == 0) {
            torch::bitwise_left_shift(input_tensor, shift_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}