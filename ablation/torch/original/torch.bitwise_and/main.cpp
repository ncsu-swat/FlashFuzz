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
        torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble || 
            input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16 ||
            input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble) {
            input = input.to(torch::kInt64);
        }
        
        if (other.dtype() == torch::kFloat || other.dtype() == torch::kDouble || 
            other.dtype() == torch::kHalf || other.dtype() == torch::kBFloat16 ||
            other.dtype() == torch::kComplexFloat || other.dtype() == torch::kComplexDouble) {
            other = other.to(torch::kInt64);
        }
        
        torch::Tensor result = torch::bitwise_and(input, other);
        
        if (offset < Size) {
            torch::Tensor out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.dtype() == torch::kFloat || out_tensor.dtype() == torch::kDouble || 
                out_tensor.dtype() == torch::kHalf || out_tensor.dtype() == torch::kBFloat16 ||
                out_tensor.dtype() == torch::kComplexFloat || out_tensor.dtype() == torch::kComplexDouble) {
                out_tensor = out_tensor.to(torch::kInt64);
            }
            
            try {
                torch::bitwise_and_out(out_tensor, input, other);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            uint8_t scalar_selector = Data[offset++];
            int64_t scalar_value = static_cast<int64_t>(scalar_selector);
            
            try {
                torch::Tensor scalar_result = torch::bitwise_and(input, scalar_value);
            } catch (...) {
            }
            
            try {
                torch::Tensor scalar_result2 = torch::bitwise_and(scalar_value, input);
            } catch (...) {
            }
        }
        
        if (input.numel() > 0 && other.numel() > 0) {
            try {
                torch::Tensor broadcasted_result = torch::bitwise_and(input, other);
            } catch (...) {
            }
        }
        
        if (input.dtype() == torch::kBool && other.dtype() == torch::kBool) {
            try {
                torch::Tensor bool_result = torch::bitwise_and(input, other);
            } catch (...) {
            }
        }
        
        if (input.numel() == 0 || other.numel() == 0) {
            try {
                torch::Tensor empty_result = torch::bitwise_and(input, other);
            } catch (...) {
            }
        }
        
        if (input.dim() == 0 && other.dim() > 0) {
            try {
                torch::Tensor scalar_broadcast = torch::bitwise_and(input, other);
            } catch (...) {
            }
        }
        
        if (input.dim() > 0 && other.dim() == 0) {
            try {
                torch::Tensor scalar_broadcast2 = torch::bitwise_and(input, other);
            } catch (...) {
            }
        }
        
        try {
            torch::Tensor inplace_input = input.clone();
            inplace_input.bitwise_and_(other);
        } catch (...) {
        }
        
        if (offset < Size) {
            try {
                torch::Tensor reshaped_input = input.view({-1});
                torch::Tensor reshaped_other = other.view({-1});
                torch::Tensor reshaped_result = torch::bitwise_and(reshaped_input, reshaped_other);
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