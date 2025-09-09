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
        
        auto result = torch::sqrt(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::sqrt(input_tensor2);
        }
        
        if (offset < Size) {
            auto input_tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::sqrt_(input_tensor3);
        }
        
        if (offset < Size) {
            auto input_tensor4 = fuzzer_utils::createTensor(Data, Size, offset);
            auto out_tensor = torch::empty_like(input_tensor4);
            torch::sqrt_out(out_tensor, input_tensor4);
        }
        
        if (offset + 1 < Size) {
            auto input_tensor5 = fuzzer_utils::createTensor(Data, Size, offset);
            uint8_t device_selector = Data[offset++];
            if (device_selector % 2 == 0 && torch::cuda::is_available()) {
                auto cuda_tensor = input_tensor5.to(torch::kCUDA);
                auto cuda_result = torch::sqrt(cuda_tensor);
            }
        }
        
        if (offset < Size) {
            auto input_tensor6 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor6.numel() > 0) {
                auto flattened = input_tensor6.flatten();
                auto sqrt_flat = torch::sqrt(flattened);
                auto reshaped = sqrt_flat.reshape(input_tensor6.sizes());
            }
        }
        
        if (offset < Size) {
            auto input_tensor7 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor7.dtype() == torch::kComplexFloat || input_tensor7.dtype() == torch::kComplexDouble) {
                auto sqrt_complex = torch::sqrt(input_tensor7);
            }
        }
        
        if (offset < Size) {
            auto input_tensor8 = fuzzer_utils::createTensor(Data, Size, offset);
            auto detached = input_tensor8.detach();
            detached.requires_grad_(true);
            auto sqrt_grad = torch::sqrt(detached);
            if (sqrt_grad.numel() > 0) {
                auto grad_output = torch::ones_like(sqrt_grad);
                sqrt_grad.backward(grad_output);
            }
        }
        
        if (offset < Size) {
            auto input_tensor9 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor9.numel() > 0) {
                auto sliced = input_tensor9.slice(0, 0, std::min(static_cast<int64_t>(2), input_tensor9.size(0)));
                auto sqrt_sliced = torch::sqrt(sliced);
            }
        }
        
        if (offset < Size) {
            auto input_tensor10 = fuzzer_utils::createTensor(Data, Size, offset);
            auto transposed = input_tensor10.t();
            auto sqrt_transposed = torch::sqrt(transposed);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}