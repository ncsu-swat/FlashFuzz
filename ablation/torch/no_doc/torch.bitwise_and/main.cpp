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
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto result = torch::bitwise_and(tensor1, tensor2);
        }
        else if (operation_mode == 1) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                int64_t scalar_raw;
                size_t scalar_bytes = sizeof(int64_t);
                if (offset + scalar_bytes <= Size) {
                    std::memcpy(&scalar_raw, Data + offset, scalar_bytes);
                    offset += scalar_bytes;
                } else {
                    scalar_raw = 42;
                }
                
                auto result = torch::bitwise_and(tensor, scalar_raw);
            }
        }
        else if (operation_mode == 2) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            torch::bitwise_and_out(tensor1, tensor1, tensor2);
        }
        else if (operation_mode == 3) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            tensor1.bitwise_and_(tensor2);
        }
        
        if (offset < Size) {
            auto extra_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                auto result1 = torch::bitwise_and(extra_tensor, base_tensor);
                auto result2 = extra_tensor & base_tensor;
                auto result3 = base_tensor.bitwise_and(extra_tensor);
            } catch (...) {
            }
        }
        
        if (offset + 1 < Size) {
            uint8_t broadcast_test = Data[offset++];
            if (broadcast_test % 2 == 0) {
                try {
                    auto small_tensor = torch::ones({1}, torch::kInt32);
                    auto large_tensor = torch::zeros({100, 100}, torch::kInt32);
                    auto broadcast_result = torch::bitwise_and(small_tensor, large_tensor);
                } catch (...) {
                }
            }
        }
        
        if (offset + 1 < Size) {
            uint8_t edge_case_test = Data[offset++];
            if (edge_case_test % 3 == 0) {
                try {
                    auto empty_tensor = torch::empty({0}, torch::kInt64);
                    auto normal_tensor = torch::ones({5}, torch::kInt64);
                    auto edge_result = torch::bitwise_and(empty_tensor, normal_tensor);
                } catch (...) {
                }
            }
        }
        
        if (offset + 1 < Size) {
            uint8_t dtype_test = Data[offset++];
            torch::ScalarType test_dtype;
            switch (dtype_test % 4) {
                case 0: test_dtype = torch::kBool; break;
                case 1: test_dtype = torch::kInt8; break;
                case 2: test_dtype = torch::kInt32; break;
                case 3: test_dtype = torch::kInt64; break;
            }
            
            try {
                auto typed_tensor1 = torch::randint(0, 256, {10, 10}, torch::TensorOptions().dtype(test_dtype));
                auto typed_tensor2 = torch::randint(0, 256, {10, 10}, torch::TensorOptions().dtype(test_dtype));
                auto typed_result = torch::bitwise_and(typed_tensor1, typed_tensor2);
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