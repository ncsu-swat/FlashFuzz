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
        
        uint8_t operation_mode = Data[offset++];
        
        if (operation_mode % 2 == 0) {
            if (offset >= Size) {
                return 0;
            }
            
            auto other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto result = torch::fmod(input_tensor, other_tensor);
            
            if (result.numel() > 0) {
                auto sum = torch::sum(result);
            }
        } else {
            if (offset + sizeof(double) > Size) {
                return 0;
            }
            
            double scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            auto result = torch::fmod(input_tensor, scalar_value);
            
            if (result.numel() > 0) {
                auto sum = torch::sum(result);
            }
        }
        
        if (offset < Size) {
            uint8_t test_edge_cases = Data[offset++];
            
            if (test_edge_cases % 4 == 0) {
                auto zero_tensor = torch::zeros_like(input_tensor);
                auto result_zero = torch::fmod(input_tensor, zero_tensor);
            } else if (test_edge_cases % 4 == 1) {
                auto result_scalar_zero = torch::fmod(input_tensor, 0.0);
            } else if (test_edge_cases % 4 == 2) {
                auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
                auto result_inf = torch::fmod(input_tensor, inf_tensor);
            } else {
                auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
                auto result_neg_inf = torch::fmod(input_tensor, neg_inf_tensor);
            }
        }
        
        if (offset < Size) {
            uint8_t broadcast_test = Data[offset++];
            
            if (broadcast_test % 3 == 0 && input_tensor.dim() > 0) {
                auto shape = input_tensor.sizes().vec();
                if (!shape.empty()) {
                    shape[0] = 1;
                    auto broadcast_tensor = torch::ones(shape, input_tensor.options());
                    auto result_broadcast = torch::fmod(input_tensor, broadcast_tensor);
                }
            } else if (broadcast_test % 3 == 1) {
                auto scalar_tensor = torch::tensor(2.5, input_tensor.options());
                auto result_scalar_tensor = torch::fmod(input_tensor, scalar_tensor);
            } else if (broadcast_test % 3 == 2 && input_tensor.numel() > 1) {
                auto reshaped = input_tensor.view({-1});
                if (reshaped.size(0) > 1) {
                    auto slice = reshaped.slice(0, 0, 1);
                    auto result_slice = torch::fmod(reshaped, slice);
                }
            }
        }
        
        if (offset < Size) {
            uint8_t type_test = Data[offset++];
            
            if (type_test % 2 == 0) {
                if (input_tensor.dtype() != torch::kBool && input_tensor.dtype() != torch::kComplexFloat && input_tensor.dtype() != torch::kComplexDouble) {
                    auto float_tensor = input_tensor.to(torch::kFloat);
                    auto result_float = torch::fmod(float_tensor, 3.14159);
                }
            } else {
                if (input_tensor.dtype().isIntegralType(false)) {
                    auto int_result = torch::fmod(input_tensor, 7);
                }
            }
        }
        
        if (offset < Size) {
            uint8_t negative_test = Data[offset++];
            
            if (negative_test % 2 == 0) {
                auto negative_input = -torch::abs(input_tensor);
                auto positive_divisor = torch::abs(input_tensor) + 1;
                auto result_neg_pos = torch::fmod(negative_input, positive_divisor);
            } else {
                auto positive_input = torch::abs(input_tensor);
                auto negative_divisor = -torch::abs(input_tensor) - 1;
                auto result_pos_neg = torch::fmod(positive_input, negative_divisor);
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