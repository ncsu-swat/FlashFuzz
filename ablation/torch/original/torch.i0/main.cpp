#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        if (Size < 2) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        auto result = torch::i0(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor out_tensor;
            try {
                out_tensor = torch::empty_like(input_tensor2);
                torch::i0_out(out_tensor, input_tensor2);
            } catch (...) {
            }
        }

        if (offset < Size) {
            auto scalar_input = fuzzer_utils::createTensor(Data, Size, offset);
            if (scalar_input.numel() > 0) {
                try {
                    auto scalar_val = scalar_input.item();
                    auto scalar_result = torch::i0(scalar_val);
                } catch (...) {
                }
            }
        }

        if (offset < Size && Size - offset >= 1) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                auto converted_input = input_tensor.to(dtype);
                auto converted_result = torch::i0(converted_input);
            } catch (...) {
            }
        }

        if (input_tensor.numel() > 0) {
            try {
                auto cloned_input = input_tensor.clone();
                auto cloned_result = torch::i0(cloned_input);
            } catch (...) {
            }
        }

        if (input_tensor.dim() > 0) {
            try {
                auto reshaped = input_tensor.flatten();
                auto reshaped_result = torch::i0(reshaped);
            } catch (...) {
            }
        }

        try {
            auto detached_input = input_tensor.detach();
            auto detached_result = torch::i0(detached_input);
        } catch (...) {
        }

        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            try {
                auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
                auto inf_result = torch::i0(inf_tensor);
            } catch (...) {
            }

            try {
                auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
                auto neg_inf_result = torch::i0(neg_inf_tensor);
            } catch (...) {
            }

            try {
                auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
                auto nan_result = torch::i0(nan_tensor);
            } catch (...) {
            }
        }

        if (input_tensor.is_floating_point()) {
            try {
                auto large_tensor = torch::full_like(input_tensor, 1e10);
                auto large_result = torch::i0(large_tensor);
            } catch (...) {
            }

            try {
                auto small_tensor = torch::full_like(input_tensor, 1e-10);
                auto small_result = torch::i0(small_tensor);
            } catch (...) {
            }

            try {
                auto negative_tensor = torch::full_like(input_tensor, -100.0);
                auto negative_result = torch::i0(negative_tensor);
            } catch (...) {
            }
        }

        if (input_tensor.numel() == 0) {
            try {
                auto empty_result = torch::i0(input_tensor);
            } catch (...) {
            }
        }

        try {
            auto contiguous_input = input_tensor.contiguous();
            auto contiguous_result = torch::i0(contiguous_input);
        } catch (...) {
        }

        if (input_tensor.dim() > 1) {
            try {
                auto transposed = input_tensor.transpose(0, 1);
                auto transposed_result = torch::i0(transposed);
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