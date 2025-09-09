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
        
        auto result = torch::abs(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::abs(input_tensor2);
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_result = torch::abs(input_tensor.item());
        }
        
        auto abs_out = torch::empty_like(input_tensor);
        torch::abs_out(abs_out, input_tensor);
        
        auto inplace_tensor = input_tensor.clone();
        inplace_tensor.abs_();
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_abs = torch::abs(input_tensor);
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_abs = torch::abs(input_tensor);
        }
        
        if (input_tensor.dim() == 0) {
            auto scalar_abs = torch::abs(input_tensor);
        }
        
        auto detached_tensor = input_tensor.detach();
        auto detached_abs = torch::abs(detached_tensor);
        
        if (input_tensor.requires_grad()) {
            auto grad_abs = torch::abs(input_tensor);
        }
        
        auto contiguous_tensor = input_tensor.contiguous();
        auto contiguous_abs = torch::abs(contiguous_tensor);
        
        auto non_contiguous = input_tensor.transpose(0, -1);
        auto non_contiguous_abs = torch::abs(non_contiguous);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}