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

        auto input = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);

        if (offset < Size) {
            double value = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&value, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else if (offset + sizeof(float) <= Size) {
                float float_val;
                std::memcpy(&float_val, Data + offset, sizeof(float));
                value = static_cast<double>(float_val);
                offset += sizeof(float);
            } else if (offset + sizeof(int32_t) <= Size) {
                int32_t int_val;
                std::memcpy(&int_val, Data + offset, sizeof(int32_t));
                value = static_cast<double>(int_val);
                offset += sizeof(int32_t);
            } else if (offset + 1 <= Size) {
                int8_t byte_val = static_cast<int8_t>(Data[offset]);
                value = static_cast<double>(byte_val);
                offset += 1;
            }

            auto result1 = torch::addcmul(input, tensor1, tensor2, value);
        }

        auto result2 = torch::addcmul(input, tensor1, tensor2);

        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                torch::addcmul_out(out_tensor, input, tensor1, tensor2);
            } catch (...) {
            }
        }

        if (input.numel() == 0 || tensor1.numel() == 0 || tensor2.numel() == 0) {
            auto empty_result = torch::addcmul(input, tensor1, tensor2);
        }

        if (input.dtype() != tensor1.dtype() || input.dtype() != tensor2.dtype()) {
            try {
                auto mixed_result = torch::addcmul(input, tensor1, tensor2);
            } catch (...) {
            }
        }

        auto zero_value_result = torch::addcmul(input, tensor1, tensor2, 0.0);
        auto negative_value_result = torch::addcmul(input, tensor1, tensor2, -1.0);
        auto large_value_result = torch::addcmul(input, tensor1, tensor2, 1e6);
        auto small_value_result = torch::addcmul(input, tensor1, tensor2, 1e-6);

        if (input.is_floating_point()) {
            auto inf_value_result = torch::addcmul(input, tensor1, tensor2, std::numeric_limits<double>::infinity());
            auto neg_inf_value_result = torch::addcmul(input, tensor1, tensor2, -std::numeric_limits<double>::infinity());
            auto nan_value_result = torch::addcmul(input, tensor1, tensor2, std::numeric_limits<double>::quiet_NaN());
        }

        if (input.sizes().size() != tensor1.sizes().size() || input.sizes().size() != tensor2.sizes().size()) {
            try {
                auto broadcast_result = torch::addcmul(input, tensor1, tensor2);
            } catch (...) {
            }
        }

        auto scalar_input = torch::scalar_tensor(1.0, input.options());
        auto scalar_tensor1 = torch::scalar_tensor(2.0, tensor1.options());
        auto scalar_tensor2 = torch::scalar_tensor(3.0, tensor2.options());
        
        try {
            auto scalar_result1 = torch::addcmul(scalar_input, tensor1, tensor2);
            auto scalar_result2 = torch::addcmul(input, scalar_tensor1, tensor2);
            auto scalar_result3 = torch::addcmul(input, tensor1, scalar_tensor2);
            auto all_scalar_result = torch::addcmul(scalar_input, scalar_tensor1, scalar_tensor2);
        } catch (...) {
        }

        if (input.is_complex() || tensor1.is_complex() || tensor2.is_complex()) {
            try {
                auto complex_result = torch::addcmul(input, tensor1, tensor2, std::complex<double>(1.0, 1.0));
            } catch (...) {
            }
        }

        auto contiguous_input = input.contiguous();
        auto non_contiguous_tensor1 = tensor1.transpose(-1, -2);
        try {
            auto non_contiguous_result = torch::addcmul(contiguous_input, non_contiguous_tensor1, tensor2);
        } catch (...) {
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}