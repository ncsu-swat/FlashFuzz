#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 3;
        
        if (operation_type == 0) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset >= Size) {
                return 0;
            }
            
            uint8_t exponent_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(exponent_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            double exponent_value;
            std::memcpy(&exponent_value, exponent_bytes, sizeof(double));
            
            auto result = torch::pow(input_tensor, exponent_value);
        }
        else if (operation_type == 1) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto exponent_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto result = torch::pow(input_tensor, exponent_tensor);
        }
        else {
            auto exponent_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset >= Size) {
                return 0;
            }
            
            uint8_t base_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(base_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            double base_value;
            std::memcpy(&base_value, base_bytes, sizeof(double));
            
            auto result = torch::pow(base_value, exponent_tensor);
        }
        
        if (offset < Size) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                uint8_t exponent_bytes[4] = {0};
                size_t bytes_to_copy = std::min(static_cast<size_t>(4), Size - offset);
                std::memcpy(exponent_bytes, Data + offset, bytes_to_copy);
                offset += bytes_to_copy;
                
                float exponent_value;
                std::memcpy(&exponent_value, exponent_bytes, sizeof(float));
                
                auto result = torch::pow(input_tensor, exponent_value);
            }
        }
        
        if (offset < Size) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                auto result = torch::pow(tensor1, tensor2);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                int32_t int_exponent;
                uint8_t int_bytes[4] = {0};
                size_t bytes_to_copy = std::min(static_cast<size_t>(4), Size - offset);
                std::memcpy(int_bytes, Data + offset, bytes_to_copy);
                std::memcpy(&int_exponent, int_bytes, sizeof(int32_t));
                offset += bytes_to_copy;
                
                auto result = torch::pow(input_tensor, int_exponent);
            }
        }
        
        if (offset < Size) {
            uint8_t negative_base_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(negative_base_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            double negative_base;
            std::memcpy(&negative_base, negative_base_bytes, sizeof(double));
            negative_base = -std::abs(negative_base);
            
            auto exponent_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                auto result = torch::pow(negative_base, exponent_tensor);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (input_tensor.numel() > 0) {
                input_tensor = input_tensor - torch::abs(input_tensor);
                
                if (offset < Size) {
                    uint8_t frac_exp_bytes[8] = {0};
                    size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
                    std::memcpy(frac_exp_bytes, Data + offset, bytes_to_copy);
                    offset += bytes_to_copy;
                    
                    double fractional_exp;
                    std::memcpy(&fractional_exp, frac_exp_bytes, sizeof(double));
                    fractional_exp = fractional_exp - std::floor(fractional_exp);
                    
                    try {
                        auto result = torch::pow(input_tensor, fractional_exp);
                    } catch (...) {
                    }
                }
            }
        }
        
        if (offset < Size) {
            auto zero_tensor = torch::zeros({2, 2});
            
            uint8_t zero_exp_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(zero_exp_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            double zero_exponent;
            std::memcpy(&zero_exponent, zero_exp_bytes, sizeof(double));
            
            auto result = torch::pow(zero_tensor, zero_exponent);
        }
        
        if (offset < Size) {
            auto large_tensor = torch::full({3, 3}, 1e10);
            
            uint8_t large_exp_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(large_exp_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            double large_exponent;
            std::memcpy(&large_exponent, large_exp_bytes, sizeof(double));
            large_exponent = std::abs(large_exponent);
            
            try {
                auto result = torch::pow(large_tensor, large_exponent);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            auto inf_tensor = torch::full({2}, std::numeric_limits<double>::infinity());
            auto nan_tensor = torch::full({2}, std::numeric_limits<double>::quiet_NaN());
            
            uint8_t special_exp_bytes[8] = {0};
            size_t bytes_to_copy = std::min(static_cast<size_t>(8), Size - offset);
            std::memcpy(special_exp_bytes, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            
            double special_exponent;
            std::memcpy(&special_exponent, special_exp_bytes, sizeof(double));
            
            try {
                auto result1 = torch::pow(inf_tensor, special_exponent);
                auto result2 = torch::pow(nan_tensor, special_exponent);
                auto result3 = torch::pow(special_exponent, inf_tensor);
                auto result4 = torch::pow(special_exponent, nan_tensor);
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