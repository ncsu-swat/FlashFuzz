#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.numel() == 0) {
            auto result = torch::atanh(input_tensor);
            return 0;
        }

        if (input_tensor.dtype() == torch::kBool) {
            input_tensor = input_tensor.to(torch::kFloat);
        }

        if (input_tensor.is_complex()) {
            auto result = torch::atanh(input_tensor);
            return 0;
        }

        if (input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kUInt8 ||
            input_tensor.dtype() == torch::kInt16 ||
            input_tensor.dtype() == torch::kInt32 ||
            input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }

        auto result = torch::atanh(input_tensor);

        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor2.numel() > 0) {
                if (input_tensor2.dtype() == torch::kBool) {
                    input_tensor2 = input_tensor2.to(torch::kFloat);
                }
                if (input_tensor2.dtype() == torch::kInt8 || 
                    input_tensor2.dtype() == torch::kUInt8 ||
                    input_tensor2.dtype() == torch::kInt16 ||
                    input_tensor2.dtype() == torch::kInt32 ||
                    input_tensor2.dtype() == torch::kInt64) {
                    input_tensor2 = input_tensor2.to(torch::kFloat);
                }
                auto result2 = torch::atanh(input_tensor2);
            }
        }

        auto cloned_input = input_tensor.clone();
        torch::atanh_(cloned_input);

        if (input_tensor.dim() > 0) {
            try {
                auto reshaped = input_tensor.reshape({-1});
                auto result_reshaped = torch::atanh(reshaped);
            } catch (...) {
            }
        }

        if (input_tensor.numel() > 1) {
            try {
                auto flattened = input_tensor.flatten();
                auto result_flat = torch::atanh(flattened);
            } catch (...) {
            }
        }

        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto result_inf = torch::atanh(inf_tensor);
            
            auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
            auto result_neg_inf = torch::atanh(neg_inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto result_nan = torch::atanh(nan_tensor);
        }

        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble ||
            input_tensor.dtype() == torch::kHalf || input_tensor.dtype() == torch::kBFloat16) {
            auto ones_tensor = torch::ones_like(input_tensor);
            auto result_ones = torch::atanh(ones_tensor);
            
            auto neg_ones_tensor = torch::full_like(input_tensor, -1.0);
            auto result_neg_ones = torch::atanh(neg_ones_tensor);
            
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto result_zero = torch::atanh(zero_tensor);
            
            auto large_tensor = torch::full_like(input_tensor, 2.0);
            auto result_large = torch::atanh(large_tensor);
            
            auto small_tensor = torch::full_like(input_tensor, 0.5);
            auto result_small = torch::atanh(small_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}