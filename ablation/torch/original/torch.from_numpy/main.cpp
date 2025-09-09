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

        auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (tensor.numel() == 0) {
            return 0;
        }

        torch::Tensor numpy_like_tensor;
        
        if (tensor.dtype() == torch::kFloat64) {
            auto data_ptr = tensor.data_ptr<double>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kFloat64);
        }
        else if (tensor.dtype() == torch::kFloat32) {
            auto data_ptr = tensor.data_ptr<float>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kFloat32);
        }
        else if (tensor.dtype() == torch::kFloat16) {
            auto data_ptr = tensor.data_ptr<torch::Half>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kFloat16);
        }
        else if (tensor.dtype() == torch::kComplexFloat) {
            auto data_ptr = tensor.data_ptr<c10::complex<float>>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kComplexFloat);
        }
        else if (tensor.dtype() == torch::kComplexDouble) {
            auto data_ptr = tensor.data_ptr<c10::complex<double>>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kComplexDouble);
        }
        else if (tensor.dtype() == torch::kInt64) {
            auto data_ptr = tensor.data_ptr<int64_t>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kInt64);
        }
        else if (tensor.dtype() == torch::kInt32) {
            auto data_ptr = tensor.data_ptr<int32_t>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kInt32);
        }
        else if (tensor.dtype() == torch::kInt16) {
            auto data_ptr = tensor.data_ptr<int16_t>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kInt16);
        }
        else if (tensor.dtype() == torch::kInt8) {
            auto data_ptr = tensor.data_ptr<int8_t>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kInt8);
        }
        else if (tensor.dtype() == torch::kUInt8) {
            auto data_ptr = tensor.data_ptr<uint8_t>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kUInt8);
        }
        else if (tensor.dtype() == torch::kBool) {
            auto data_ptr = tensor.data_ptr<bool>();
            numpy_like_tensor = torch::from_blob(data_ptr, tensor.sizes(), torch::kBool);
        }
        else {
            return 0;
        }

        auto result_tensor = numpy_like_tensor.clone();
        
        if (result_tensor.numel() > 0) {
            auto sum_result = torch::sum(result_tensor);
            auto mean_result = torch::mean(result_tensor.to(torch::kFloat));
            
            if (result_tensor.dim() > 0) {
                auto reshaped = result_tensor.view({-1});
                if (reshaped.numel() > 1) {
                    auto sorted = torch::sort(reshaped);
                }
            }
        }

        if (offset < Size) {
            auto second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (second_tensor.numel() > 0 && second_tensor.dtype() == tensor.dtype()) {
                if (second_tensor.sizes() == tensor.sizes()) {
                    auto combined = torch::stack({result_tensor, second_tensor});
                    auto selected = torch::select(combined, 0, 0);
                }
            }
        }

        if (result_tensor.dim() > 1) {
            auto transposed = result_tensor.transpose(0, -1);
            auto contiguous = transposed.contiguous();
        }

        if (result_tensor.numel() > 4) {
            auto sliced = result_tensor.slice(0, 0, std::min(static_cast<int64_t>(2), result_tensor.size(0)));
        }

        auto detached = result_tensor.detach();
        auto requires_grad_tensor = result_tensor.requires_grad_(false);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}