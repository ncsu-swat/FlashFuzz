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

            auto result = torch::addcdiv(input, tensor1, tensor2, value);
        } else {
            auto result = torch::addcdiv(input, tensor1, tensor2);
        }

        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                torch::addcdiv_out(out_tensor, input, tensor1, tensor2);
            } catch (...) {
            }
        }

        if (input.numel() == 0 || tensor1.numel() == 0 || tensor2.numel() == 0) {
            auto result = torch::addcdiv(input, tensor1, tensor2, 0.0);
        }

        if (tensor2.dtype() == torch::kFloat || tensor2.dtype() == torch::kDouble) {
            auto zero_tensor = torch::zeros_like(tensor2);
            try {
                auto result = torch::addcdiv(input, tensor1, zero_tensor, 1.0);
            } catch (...) {
            }
        }

        if (input.dtype() == torch::kInt32 || input.dtype() == torch::kInt64) {
            try {
                auto result = torch::addcdiv(input, tensor1, tensor2, 2);
            } catch (...) {
            }
        }

        auto large_value = std::numeric_limits<double>::max();
        try {
            auto result = torch::addcdiv(input, tensor1, tensor2, large_value);
        } catch (...) {
        }

        auto small_value = std::numeric_limits<double>::min();
        try {
            auto result = torch::addcdiv(input, tensor1, tensor2, small_value);
        } catch (...) {
        }

        try {
            auto result = torch::addcdiv(input, tensor1, tensor2, -1.0);
        } catch (...) {
        }

        if (tensor1.dtype() == torch::kComplexFloat || tensor1.dtype() == torch::kComplexDouble) {
            try {
                auto result = torch::addcdiv(input, tensor1, tensor2, std::complex<double>(1.0, 1.0));
            } catch (...) {
            }
        }

        auto inf_value = std::numeric_limits<double>::infinity();
        try {
            auto result = torch::addcdiv(input, tensor1, tensor2, inf_value);
        } catch (...) {
        }

        auto neg_inf_value = -std::numeric_limits<double>::infinity();
        try {
            auto result = torch::addcdiv(input, tensor1, tensor2, neg_inf_value);
        } catch (...) {
        }

        auto nan_value = std::numeric_limits<double>::quiet_NaN();
        try {
            auto result = torch::addcdiv(input, tensor1, tensor2, nan_value);
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