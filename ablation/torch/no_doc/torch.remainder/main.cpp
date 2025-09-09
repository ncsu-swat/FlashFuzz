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
        
        auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t operation_mode = Data[offset++];
        
        if (operation_mode % 2 == 0) {
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::remainder(tensor1, tensor2);
        } else {
            if (offset + sizeof(double) <= Size) {
                double scalar_value;
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                torch::remainder(tensor1, scalar_value);
            } else {
                torch::remainder(tensor1, 1.0);
            }
        }
        
        if (offset < Size) {
            uint8_t inplace_mode = Data[offset++];
            if (inplace_mode % 3 == 0) {
                auto tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
                if (offset < Size) {
                    auto tensor4 = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor3.remainder_(tensor4);
                }
            } else if (inplace_mode % 3 == 1) {
                auto tensor5 = fuzzer_utils::createTensor(Data, Size, offset);
                if (offset + sizeof(float) <= Size) {
                    float scalar_val;
                    std::memcpy(&scalar_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    tensor5.remainder_(scalar_val);
                }
            }
        }
        
        if (offset < Size) {
            auto dividend = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset < Size) {
                auto divisor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = torch::empty_like(dividend);
                torch::remainder_out(result, dividend, divisor);
            }
        }
        
        if (offset < Size) {
            auto tensor6 = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset + sizeof(int64_t) <= Size) {
                int64_t int_scalar;
                std::memcpy(&int_scalar, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                torch::remainder(tensor6, int_scalar);
            }
        }
        
        if (offset < Size) {
            auto tensor7 = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset + sizeof(int32_t) <= Size) {
                int32_t int32_scalar;
                std::memcpy(&int32_scalar, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                torch::remainder(tensor7, int32_scalar);
            }
        }
        
        if (offset < Size) {
            auto tensor8 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::remainder(tensor8, 0.0);
        }
        
        if (offset < Size) {
            auto tensor9 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::remainder(tensor9, std::numeric_limits<double>::infinity());
        }
        
        if (offset < Size) {
            auto tensor10 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::remainder(tensor10, -std::numeric_limits<double>::infinity());
        }
        
        if (offset < Size) {
            auto tensor11 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::remainder(tensor11, std::numeric_limits<double>::quiet_NaN());
        }
        
        if (offset < Size) {
            auto tensor12 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::remainder(tensor12, std::numeric_limits<double>::min());
        }
        
        if (offset < Size) {
            auto tensor13 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::remainder(tensor13, std::numeric_limits<double>::max());
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}