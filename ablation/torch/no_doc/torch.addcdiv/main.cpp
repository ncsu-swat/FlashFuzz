#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        if (offset < Size) {
            uint8_t value_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(value_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            double value;
            std::memcpy(&value, value_bytes, sizeof(double));
            
            torch::addcdiv_out(input_tensor, input_tensor, tensor1, tensor2, value);
        }

        if (offset < Size) {
            uint8_t value_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(value_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            float value;
            std::memcpy(&value, value_bytes, sizeof(float));
            
            auto result1 = torch::addcdiv(input_tensor, tensor1, tensor2, value);
        }

        if (offset < Size) {
            uint8_t value_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(value_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            int64_t value;
            std::memcpy(&value, value_bytes, sizeof(int64_t));
            
            auto result2 = torch::addcdiv(input_tensor, tensor1, tensor2, value);
        }

        auto result3 = torch::addcdiv(input_tensor, tensor1, tensor2);

        if (offset < Size) {
            uint8_t value_bytes[4] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(4), Size - offset);
            std::memcpy(value_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            int32_t value;
            std::memcpy(&value, value_bytes, sizeof(int32_t));
            
            auto result4 = torch::addcdiv(input_tensor, tensor1, tensor2, value);
        }

        auto scalar_tensor = torch::scalar_tensor(1.5);
        auto result5 = torch::addcdiv(input_tensor, tensor1, tensor2, scalar_tensor);

        if (offset < Size) {
            uint8_t value_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(value_bytes, Data + offset, bytes_to_copy);
            
            double extreme_value;
            std::memcpy(&extreme_value, value_bytes, sizeof(double));
            
            auto result6 = torch::addcdiv(input_tensor, tensor1, tensor2, extreme_value);
        }

        auto zero_tensor = torch::zeros_like(tensor2);
        auto result7 = torch::addcdiv(input_tensor, tensor1, zero_tensor, 1.0);

        auto inf_tensor = torch::full_like(tensor2, std::numeric_limits<double>::infinity());
        auto result8 = torch::addcdiv(input_tensor, tensor1, inf_tensor, 1.0);

        auto nan_tensor = torch::full_like(tensor2, std::numeric_limits<double>::quiet_NaN());
        auto result9 = torch::addcdiv(input_tensor, tensor1, nan_tensor, 1.0);

        auto very_small_tensor = torch::full_like(tensor2, 1e-20);
        auto result10 = torch::addcdiv(input_tensor, tensor1, very_small_tensor, 1.0);

        auto very_large_tensor = torch::full_like(tensor2, 1e20);
        auto result11 = torch::addcdiv(input_tensor, tensor1, very_large_tensor, 1.0);

        auto negative_tensor = torch::full_like(tensor2, -1.0);
        auto result12 = torch::addcdiv(input_tensor, tensor1, negative_tensor, 1.0);

        if (input_tensor.numel() > 0 && tensor1.numel() > 0 && tensor2.numel() > 0) {
            try {
                auto broadcasted_result = torch::addcdiv(input_tensor, tensor1, tensor2, 2.5);
            } catch (...) {
            }
        }

        auto empty_tensor = torch::empty({0});
        try {
            auto empty_result = torch::addcdiv(empty_tensor, empty_tensor, empty_tensor, 1.0);
        } catch (...) {
        }

        if (offset < Size) {
            auto complex_input = input_tensor.to(torch::kComplexFloat);
            auto complex_t1 = tensor1.to(torch::kComplexFloat);
            auto complex_t2 = tensor2.to(torch::kComplexFloat);
            
            try {
                auto complex_result = torch::addcdiv(complex_input, complex_t1, complex_t2, 1.0);
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