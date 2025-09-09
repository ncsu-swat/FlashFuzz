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

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        auto result = torch::sign(input_tensor);

        if (offset < Size) {
            uint8_t out_flag = Data[offset++];
            if (out_flag % 2 == 1) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                try {
                    torch::sign_out(out_tensor, input_tensor);
                } catch (...) {
                }
            }
        }

        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                auto result2 = torch::sign(input_tensor2);
            } catch (...) {
            }
        }

        auto zero_tensor = torch::zeros({1});
        auto zero_result = torch::sign(zero_tensor);

        auto inf_tensor = torch::full({1}, std::numeric_limits<float>::infinity());
        auto inf_result = torch::sign(inf_tensor);

        auto neg_inf_tensor = torch::full({1}, -std::numeric_limits<float>::infinity());
        auto neg_inf_result = torch::sign(neg_inf_tensor);

        auto nan_tensor = torch::full({1}, std::numeric_limits<float>::quiet_NaN());
        auto nan_result = torch::sign(nan_tensor);

        auto empty_tensor = torch::empty({0});
        auto empty_result = torch::sign(empty_tensor);

        auto scalar_tensor = torch::tensor(3.14);
        auto scalar_result = torch::sign(scalar_tensor);

        auto neg_scalar_tensor = torch::tensor(-2.71);
        auto neg_scalar_result = torch::sign(neg_scalar_tensor);

        if (input_tensor.numel() > 0) {
            auto inplace_tensor = input_tensor.clone();
            inplace_tensor.sign_();
        }

        auto complex_tensor = torch::tensor({{1.0, 2.0}, {-3.0, 4.0}}, torch::kComplexFloat);
        auto complex_result = torch::sign(complex_tensor);

        auto bool_tensor = torch::tensor({true, false, true}, torch::kBool);
        auto bool_result = torch::sign(bool_tensor);

        auto int_tensor = torch::tensor({-5, 0, 7, -1}, torch::kInt32);
        auto int_result = torch::sign(int_tensor);

        auto large_tensor = torch::full({1000, 1000}, 1e10);
        auto large_result = torch::sign(large_tensor);

        auto small_tensor = torch::full({1000, 1000}, 1e-10);
        auto small_result = torch::sign(small_tensor);

        auto mixed_tensor = torch::tensor({-1e20, 0.0, 1e-20, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
        auto mixed_result = torch::sign(mixed_tensor);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}