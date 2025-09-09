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
        
        auto other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.dtype() != torch::kInt8 && 
            input_tensor.dtype() != torch::kUInt8 &&
            input_tensor.dtype() != torch::kInt16 &&
            input_tensor.dtype() != torch::kInt32 &&
            input_tensor.dtype() != torch::kInt64 &&
            input_tensor.dtype() != torch::kBool) {
            
            input_tensor = input_tensor.to(torch::kInt32);
        }
        
        if (other_tensor.dtype() != torch::kInt8 && 
            other_tensor.dtype() != torch::kUInt8 &&
            other_tensor.dtype() != torch::kInt16 &&
            other_tensor.dtype() != torch::kInt32 &&
            other_tensor.dtype() != torch::kInt64 &&
            other_tensor.dtype() != torch::kBool) {
            
            other_tensor = other_tensor.to(torch::kInt32);
        }
        
        torch::Tensor result;
        
        if (input_tensor.sizes() == other_tensor.sizes()) {
            result = torch::bitwise_xor(input_tensor, other_tensor);
        } else {
            try {
                result = torch::bitwise_xor(input_tensor, other_tensor);
            } catch (...) {
                if (input_tensor.numel() == 1) {
                    auto scalar_input = input_tensor.item();
                    result = torch::bitwise_xor(scalar_input, other_tensor);
                } else if (other_tensor.numel() == 1) {
                    auto scalar_other = other_tensor.item();
                    result = torch::bitwise_xor(input_tensor, scalar_other);
                } else {
                    throw;
                }
            }
        }
        
        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.dtype() == result.dtype() && out_tensor.sizes() == result.sizes()) {
                torch::bitwise_xor_out(out_tensor, input_tensor, other_tensor);
            }
        }
        
        auto empty_input = torch::empty({0}, input_tensor.options());
        auto empty_other = torch::empty({0}, other_tensor.options());
        torch::bitwise_xor(empty_input, empty_other);
        
        auto zero_dim_input = torch::tensor(42, input_tensor.options());
        auto zero_dim_other = torch::tensor(13, other_tensor.options());
        torch::bitwise_xor(zero_dim_input, zero_dim_other);
        
        if (input_tensor.dtype() == torch::kBool && other_tensor.dtype() == torch::kBool) {
            auto bool_result = torch::bitwise_xor(input_tensor, other_tensor);
        }
        
        if (input_tensor.numel() > 0 && other_tensor.numel() > 0) {
            auto large_input = input_tensor.expand({std::max(input_tensor.size(0), 1000L)});
            auto large_other = other_tensor.expand({std::max(other_tensor.size(0), 1000L)});
            if (large_input.sizes() == large_other.sizes()) {
                torch::bitwise_xor(large_input, large_other);
            }
        }
        
        auto neg_input = -torch::abs(input_tensor);
        auto neg_other = -torch::abs(other_tensor);
        if (neg_input.sizes() == neg_other.sizes()) {
            torch::bitwise_xor(neg_input, neg_other);
        }
        
        auto max_val_tensor = torch::full_like(input_tensor, std::numeric_limits<int32_t>::max());
        auto min_val_tensor = torch::full_like(other_tensor, std::numeric_limits<int32_t>::min());
        if (max_val_tensor.sizes() == min_val_tensor.sizes()) {
            torch::bitwise_xor(max_val_tensor, min_val_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}