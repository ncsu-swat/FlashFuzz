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
        
        uint8_t operation_mode = Data[offset++];
        operation_mode = operation_mode % 4;
        
        if (operation_mode == 0) {
            auto base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto exponent_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto result = torch::pow(base_tensor, exponent_tensor);
        }
        else if (operation_mode == 1) {
            auto base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset + sizeof(double) <= Size) {
                double exponent_scalar;
                std::memcpy(&exponent_scalar, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                auto result = torch::pow(base_tensor, exponent_scalar);
            }
        }
        else if (operation_mode == 2) {
            if (offset + sizeof(double) <= Size) {
                double base_scalar;
                std::memcpy(&base_scalar, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                auto exponent_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                auto result = torch::pow(base_scalar, exponent_tensor);
            }
        }
        else if (operation_mode == 3) {
            auto base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset + sizeof(int64_t) <= Size) {
                int64_t exponent_int;
                std::memcpy(&exponent_int, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                auto result = torch::pow(base_tensor, exponent_int);
            }
        }
        
        if (offset < Size) {
            uint8_t inplace_mode = Data[offset++];
            if (inplace_mode % 2 == 1 && offset < Size) {
                auto base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto exponent_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                base_tensor.pow_(exponent_tensor);
            }
        }
        
        if (offset < Size) {
            uint8_t out_mode = Data[offset++];
            if (out_mode % 2 == 1 && offset < Size) {
                auto base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto exponent_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                torch::pow_out(out_tensor, base_tensor, exponent_tensor);
            }
        }
        
        if (offset < Size) {
            uint8_t special_case = Data[offset++];
            special_case = special_case % 8;
            
            if (special_case == 0) {
                auto zero_tensor = torch::zeros({2, 3});
                auto result = torch::pow(zero_tensor, 0.0);
            }
            else if (special_case == 1) {
                auto ones_tensor = torch::ones({2, 3});
                auto result = torch::pow(ones_tensor, std::numeric_limits<double>::infinity());
            }
            else if (special_case == 2) {
                auto neg_tensor = torch::full({2, 3}, -1.0);
                auto result = torch::pow(neg_tensor, 0.5);
            }
            else if (special_case == 3) {
                auto large_tensor = torch::full({2, 3}, 1e10);
                auto result = torch::pow(large_tensor, 2.0);
            }
            else if (special_case == 4) {
                auto small_tensor = torch::full({2, 3}, 1e-10);
                auto result = torch::pow(small_tensor, -2.0);
            }
            else if (special_case == 5) {
                auto inf_tensor = torch::full({2, 3}, std::numeric_limits<double>::infinity());
                auto result = torch::pow(inf_tensor, 0.5);
            }
            else if (special_case == 6) {
                auto nan_tensor = torch::full({2, 3}, std::numeric_limits<double>::quiet_NaN());
                auto result = torch::pow(nan_tensor, 2.0);
            }
            else if (special_case == 7) {
                auto empty_tensor = torch::empty({0});
                auto result = torch::pow(empty_tensor, 2.0);
            }
        }
        
        if (offset < Size) {
            uint8_t complex_mode = Data[offset++];
            if (complex_mode % 2 == 1) {
                auto complex_tensor = torch::randn({2, 2}, torch::dtype(torch::kComplexFloat));
                auto result = torch::pow(complex_tensor, 2.0);
                
                auto complex_base = torch::randn({2, 2}, torch::dtype(torch::kComplexFloat));
                auto complex_exp = torch::randn({2, 2}, torch::dtype(torch::kComplexFloat));
                auto complex_result = torch::pow(complex_base, complex_exp);
            }
        }
        
        if (offset < Size) {
            uint8_t broadcast_mode = Data[offset++];
            if (broadcast_mode % 2 == 1) {
                auto base_tensor = torch::randn({3, 1, 4});
                auto exp_tensor = torch::randn({1, 2, 1});
                auto result = torch::pow(base_tensor, exp_tensor);
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