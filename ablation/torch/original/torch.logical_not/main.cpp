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

        torch::Tensor result = torch::logical_not(input_tensor);

        if (offset < Size) {
            uint8_t out_selector = Data[offset++];
            if (out_selector % 2 == 1 && offset < Size) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor result_with_out = torch::logical_not(input_tensor, out_tensor);
            }
        }

        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto out_dtype = fuzzer_utils::parseDataType(dtype_selector);
            auto out_tensor = torch::empty_like(input_tensor, torch::TensorOptions().dtype(out_dtype));
            torch::Tensor result_typed_out = torch::logical_not(input_tensor, out_tensor);
        }

        if (offset < Size) {
            uint8_t shape_modifier = Data[offset++];
            if (shape_modifier % 3 == 0 && input_tensor.numel() > 0) {
                auto reshaped = input_tensor.view({-1});
                torch::Tensor result_reshaped = torch::logical_not(reshaped);
            }
        }

        if (offset < Size && input_tensor.dim() > 0) {
            uint8_t slice_selector = Data[offset++];
            int64_t slice_dim = slice_selector % input_tensor.dim();
            if (input_tensor.size(slice_dim) > 1) {
                auto sliced = input_tensor.slice(slice_dim, 0, 1);
                torch::Tensor result_sliced = torch::logical_not(sliced);
            }
        }

        if (input_tensor.numel() == 0) {
            torch::Tensor result_empty = torch::logical_not(input_tensor);
        }

        if (input_tensor.dtype() == torch::kBool) {
            torch::Tensor result_bool = torch::logical_not(input_tensor);
        }

        if (input_tensor.dtype().isFloatingPoint()) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            torch::Tensor result_inf = torch::logical_not(inf_tensor);
            
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            torch::Tensor result_nan = torch::logical_not(nan_tensor);
        }

        if (input_tensor.dtype().isIntegral(false)) {
            auto max_tensor = torch::full_like(input_tensor, std::numeric_limits<int64_t>::max());
            torch::Tensor result_max = torch::logical_not(max_tensor);
            
            auto min_tensor = torch::full_like(input_tensor, std::numeric_limits<int64_t>::min());
            torch::Tensor result_min = torch::logical_not(min_tensor);
        }

        if (input_tensor.dtype().isComplex()) {
            torch::Tensor result_complex = torch::logical_not(input_tensor);
        }

        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            if (device_selector % 4 == 0 && torch::cuda::is_available()) {
                auto cuda_tensor = input_tensor.to(torch::kCUDA);
                torch::Tensor result_cuda = torch::logical_not(cuda_tensor);
            }
        }

        auto zero_tensor = torch::zeros_like(input_tensor);
        torch::Tensor result_zeros = torch::logical_not(zero_tensor);

        auto ones_tensor = torch::ones_like(input_tensor);
        torch::Tensor result_ones = torch::logical_not(ones_tensor);

        if (input_tensor.requires_grad()) {
            torch::Tensor result_grad = torch::logical_not(input_tensor);
        }

        if (input_tensor.is_sparse()) {
            torch::Tensor result_sparse = torch::logical_not(input_tensor);
        }

        if (offset < Size) {
            uint8_t memory_format_selector = Data[offset++];
            if (memory_format_selector % 2 == 1 && input_tensor.dim() >= 4) {
                auto channels_last = input_tensor.to(torch::MemoryFormat::ChannelsLast);
                torch::Tensor result_channels_last = torch::logical_not(channels_last);
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