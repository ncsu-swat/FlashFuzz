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

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        auto other_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        torch::logical_and(input_tensor, other_tensor);

        if (offset < Size) {
            uint8_t use_out_param = Data[offset++];
            if (use_out_param % 2 == 1) {
                auto out_tensor = torch::empty_like(input_tensor, torch::TensorOptions().dtype(torch::kBool));
                torch::logical_and_out(out_tensor, input_tensor, other_tensor);
            }
        }

        if (offset < Size) {
            uint8_t broadcast_test = Data[offset++];
            if (broadcast_test % 3 == 0) {
                auto scalar_tensor = torch::tensor(static_cast<bool>(broadcast_test % 2));
                torch::logical_and(input_tensor, scalar_tensor);
                torch::logical_and(scalar_tensor, other_tensor);
            }
        }

        if (offset < Size) {
            uint8_t empty_test = Data[offset++];
            if (empty_test % 4 == 0) {
                auto empty_tensor = torch::empty({0}, input_tensor.options());
                torch::logical_and(empty_tensor, empty_tensor);
            }
        }

        if (offset < Size) {
            uint8_t mixed_dtype_test = Data[offset++];
            if (mixed_dtype_test % 5 == 0) {
                auto int_tensor = input_tensor.to(torch::kInt32);
                auto float_tensor = other_tensor.to(torch::kFloat32);
                torch::logical_and(int_tensor, float_tensor);
            }
        }

        if (offset < Size) {
            uint8_t zero_dim_test = Data[offset++];
            if (zero_dim_test % 6 == 0) {
                auto zero_dim = torch::tensor(42.0);
                torch::logical_and(zero_dim, input_tensor);
            }
        }

        if (offset < Size) {
            uint8_t large_tensor_test = Data[offset++];
            if (large_tensor_test % 7 == 0) {
                try {
                    auto large_shape = std::vector<int64_t>{1000, 1000};
                    auto large_tensor = torch::zeros(large_shape, input_tensor.options());
                    torch::logical_and(large_tensor, large_tensor);
                } catch (...) {
                }
            }
        }

        if (offset < Size) {
            uint8_t complex_test = Data[offset++];
            if (complex_test % 8 == 0) {
                try {
                    auto complex_tensor = input_tensor.to(torch::kComplexFloat);
                    torch::logical_and(complex_tensor, complex_tensor);
                } catch (...) {
                }
            }
        }

        if (offset < Size) {
            uint8_t inf_nan_test = Data[offset++];
            if (inf_nan_test % 9 == 0) {
                auto inf_tensor = torch::full_like(input_tensor.to(torch::kFloat), std::numeric_limits<float>::infinity());
                auto nan_tensor = torch::full_like(input_tensor.to(torch::kFloat), std::numeric_limits<float>::quiet_NaN());
                torch::logical_and(inf_tensor, nan_tensor);
                torch::logical_and(nan_tensor, inf_tensor);
            }
        }

        if (offset < Size) {
            uint8_t negative_test = Data[offset++];
            if (negative_test % 10 == 0) {
                auto neg_tensor = -torch::abs(input_tensor.to(torch::kFloat));
                torch::logical_and(neg_tensor, other_tensor);
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