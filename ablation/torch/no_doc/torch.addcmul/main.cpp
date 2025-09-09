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

        float value = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }

        torch::addcmul(input_tensor, tensor1, tensor2);

        torch::addcmul(input_tensor, tensor1, tensor2, value);

        auto result1 = torch::addcmul(input_tensor, tensor1, tensor2);
        auto result2 = torch::addcmul(input_tensor, tensor1, tensor2, value);

        input_tensor.addcmul_(tensor1, tensor2);
        input_tensor.addcmul_(tensor1, tensor2, value);

        auto empty_tensor = torch::empty({0});
        torch::addcmul(empty_tensor, empty_tensor, empty_tensor);

        auto scalar_tensor = torch::tensor(1.0f);
        torch::addcmul(scalar_tensor, scalar_tensor, scalar_tensor);

        if (input_tensor.numel() > 0 && tensor1.numel() > 0 && tensor2.numel() > 0) {
            try {
                auto broadcasted_result = torch::addcmul(input_tensor, tensor1, tensor2);
            } catch (...) {
            }
        }

        if (offset < Size) {
            double double_value = static_cast<double>(value);
            torch::addcmul(input_tensor, tensor1, tensor2, double_value);
        }

        if (offset < Size) {
            int int_value = static_cast<int>(value);
            torch::addcmul(input_tensor, tensor1, tensor2, int_value);
        }

        auto complex_input = input_tensor.to(torch::kComplexFloat);
        auto complex_t1 = tensor1.to(torch::kComplexFloat);
        auto complex_t2 = tensor2.to(torch::kComplexFloat);
        torch::addcmul(complex_input, complex_t1, complex_t2);

        auto bool_input = input_tensor.to(torch::kBool);
        auto bool_t1 = tensor1.to(torch::kBool);
        auto bool_t2 = tensor2.to(torch::kBool);
        torch::addcmul(bool_input, bool_t1, bool_t2);

        auto int_input = input_tensor.to(torch::kInt64);
        auto int_t1 = tensor1.to(torch::kInt64);
        auto int_t2 = tensor2.to(torch::kInt64);
        torch::addcmul(int_input, int_t1, int_t2);

        torch::addcmul(input_tensor, tensor1, tensor2, std::numeric_limits<float>::infinity());
        torch::addcmul(input_tensor, tensor1, tensor2, -std::numeric_limits<float>::infinity());
        torch::addcmul(input_tensor, tensor1, tensor2, std::numeric_limits<float>::quiet_NaN());
        torch::addcmul(input_tensor, tensor1, tensor2, std::numeric_limits<float>::max());
        torch::addcmul(input_tensor, tensor1, tensor2, std::numeric_limits<float>::lowest());
        torch::addcmul(input_tensor, tensor1, tensor2, 0.0f);
        torch::addcmul(input_tensor, tensor1, tensor2, -0.0f);

        if (input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view({-1});
            torch::addcmul(reshaped, tensor1, tensor2);
        }

        auto out_tensor = torch::empty_like(input_tensor);
        torch::addcmul_out(out_tensor, input_tensor, tensor1, tensor2);
        torch::addcmul_out(out_tensor, input_tensor, tensor1, tensor2, value);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}