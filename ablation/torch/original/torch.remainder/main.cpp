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
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto result = torch::remainder(input_tensor, other_tensor);
        }
        else if (operation_mode == 1) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                double scalar_value;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&scalar_value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                } else {
                    scalar_value = static_cast<double>(Data[offset++]);
                }
                
                auto result = torch::remainder(input_tensor, scalar_value);
            }
        }
        else if (operation_mode == 2) {
            if (offset + sizeof(double) <= Size) {
                double scalar_input;
                std::memcpy(&scalar_input, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                auto other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                auto result = torch::remainder(scalar_input, other_tensor);
            }
        }
        else if (operation_mode == 3) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto out_tensor = torch::empty_like(input_tensor);
            torch::remainder_out(out_tensor, input_tensor, other_tensor);
        }
        
        if (offset < Size) {
            uint8_t edge_case_mode = Data[offset++];
            edge_case_mode = edge_case_mode % 8;
            
            if (edge_case_mode == 0) {
                auto zero_tensor = torch::zeros({2, 3});
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = torch::remainder(input_tensor, zero_tensor);
            }
            else if (edge_case_mode == 1) {
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = torch::remainder(input_tensor, 0.0);
            }
            else if (edge_case_mode == 2) {
                auto inf_tensor = torch::full({2, 2}, std::numeric_limits<double>::infinity());
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = torch::remainder(input_tensor, inf_tensor);
            }
            else if (edge_case_mode == 3) {
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto nan_tensor = torch::full({1}, std::numeric_limits<double>::quiet_NaN());
                auto result = torch::remainder(input_tensor, nan_tensor);
            }
            else if (edge_case_mode == 4) {
                auto neg_tensor = torch::full({3, 3}, -2.5);
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = torch::remainder(input_tensor, neg_tensor);
            }
            else if (edge_case_mode == 5) {
                auto very_small = torch::full({2}, 1e-10);
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = torch::remainder(input_tensor, very_small);
            }
            else if (edge_case_mode == 6) {
                auto very_large = torch::full({2}, 1e10);
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = torch::remainder(input_tensor, very_large);
            }
            else if (edge_case_mode == 7) {
                auto empty_tensor = torch::empty({0});
                auto other_empty = torch::empty({0});
                auto result = torch::remainder(empty_tensor, other_empty);
            }
        }
        
        if (offset < Size) {
            uint8_t broadcast_mode = Data[offset++];
            broadcast_mode = broadcast_mode % 4;
            
            if (broadcast_mode == 0) {
                auto tensor1 = torch::randn({1, 3});
                auto tensor2 = torch::randn({4, 1});
                auto result = torch::remainder(tensor1, tensor2);
            }
            else if (broadcast_mode == 1) {
                auto tensor1 = torch::randn({2, 1, 3});
                auto tensor2 = torch::randn({1, 4, 1});
                auto result = torch::remainder(tensor1, tensor2);
            }
            else if (broadcast_mode == 2) {
                auto scalar_tensor = torch::randn({});
                auto regular_tensor = torch::randn({3, 3});
                auto result = torch::remainder(scalar_tensor, regular_tensor);
            }
            else if (broadcast_mode == 3) {
                auto tensor1 = torch::randn({5});
                auto tensor2 = torch::randn({1, 5});
                auto result = torch::remainder(tensor1, tensor2);
            }
        }
        
        if (offset < Size) {
            uint8_t dtype_mode = Data[offset++];
            dtype_mode = dtype_mode % 6;
            
            if (dtype_mode == 0) {
                auto int_tensor = torch::randint(-10, 10, {3, 3}, torch::kInt32);
                auto float_tensor = torch::randn({3, 3}, torch::kFloat32);
                auto result = torch::remainder(int_tensor, float_tensor);
            }
            else if (dtype_mode == 1) {
                auto double_tensor = torch::randn({2, 2}, torch::kDouble);
                auto int_tensor = torch::randint(1, 5, {2, 2}, torch::kInt64);
                auto result = torch::remainder(double_tensor, int_tensor);
            }
            else if (dtype_mode == 2) {
                auto bool_tensor = torch::randint(0, 2, {2, 2}, torch::kBool);
                auto float_tensor = torch::randn({2, 2}, torch::kFloat32);
                auto result = torch::remainder(bool_tensor, float_tensor);
            }
            else if (dtype_mode == 3) {
                auto half_tensor = torch::randn({2, 2}, torch::kHalf);
                auto double_tensor = torch::randn({2, 2}, torch::kDouble);
                auto result = torch::remainder(half_tensor, double_tensor);
            }
            else if (dtype_mode == 4) {
                auto uint8_tensor = torch::randint(0, 255, {3, 3}, torch::kUInt8);
                auto int8_tensor = torch::randint(-128, 127, {3, 3}, torch::kInt8);
                auto result = torch::remainder(uint8_tensor, int8_tensor);
            }
            else if (dtype_mode == 5) {
                auto bfloat16_tensor = torch::randn({2, 2}, torch::kBFloat16);
                auto float_tensor = torch::randn({2, 2}, torch::kFloat32);
                auto result = torch::remainder(bfloat16_tensor, float_tensor);
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