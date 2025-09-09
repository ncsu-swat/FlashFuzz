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
        
        auto result = torch::real(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::real(input_tensor2);
        }
        
        if (input_tensor.is_complex()) {
            auto real_part = torch::real(input_tensor);
            auto imag_part = torch::imag(input_tensor);
            auto reconstructed = torch::complex(real_part, imag_part);
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_real = torch::real(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        auto ones_real = torch::real(ones_tensor);
        
        if (input_tensor.numel() > 0) {
            auto sliced = input_tensor.slice(0, 0, std::min(static_cast<int64_t>(1), input_tensor.size(0)));
            auto sliced_real = torch::real(sliced);
        }
        
        if (input_tensor.dim() > 1) {
            auto reshaped = input_tensor.reshape({-1});
            auto reshaped_real = torch::real(reshaped);
        }
        
        auto cloned = input_tensor.clone();
        auto cloned_real = torch::real(cloned);
        
        if (input_tensor.is_cuda()) {
            auto cpu_tensor = input_tensor.cpu();
            auto cpu_real = torch::real(cpu_tensor);
        }
        
        auto detached = input_tensor.detach();
        auto detached_real = torch::real(detached);
        
        if (input_tensor.requires_grad()) {
            auto no_grad_tensor = input_tensor.detach();
            auto no_grad_real = torch::real(no_grad_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}