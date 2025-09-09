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
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t conj_flag = Data[offset++];
        bool should_conjugate = (conj_flag % 2) == 1;
        
        torch::Tensor test_tensor = input_tensor;
        
        if (should_conjugate && (input_tensor.dtype() == torch::kComplexFloat || 
                                input_tensor.dtype() == torch::kComplexDouble)) {
            test_tensor = input_tensor.conj();
        }
        
        auto result = torch::resolve_conj(test_tensor);
        
        if (test_tensor.is_conj()) {
            if (result.is_conj()) {
                throw std::runtime_error("resolve_conj should clear conjugate bit");
            }
        }
        
        if (input_tensor.numel() == 0) {
            torch::resolve_conj(input_tensor);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || 
            input_tensor.dtype() == torch::kComplexDouble) {
            auto conj_tensor = input_tensor.conj();
            auto resolved = torch::resolve_conj(conj_tensor);
            torch::resolve_conj(resolved);
        }
        
        auto cloned = input_tensor.clone();
        torch::resolve_conj(cloned);
        
        if (input_tensor.dim() > 0) {
            auto view = input_tensor.view({-1});
            torch::resolve_conj(view);
        }
        
        if (input_tensor.numel() > 1) {
            auto slice = input_tensor.slice(0, 0, 1);
            torch::resolve_conj(slice);
        }
        
        auto detached = input_tensor.detach();
        torch::resolve_conj(detached);
        
        if (offset < Size) {
            uint8_t device_flag = Data[offset++];
            if (device_flag % 4 == 0 && torch::cuda::is_available()) {
                try {
                    auto cuda_tensor = input_tensor.to(torch::kCUDA);
                    torch::resolve_conj(cuda_tensor);
                } catch (...) {
                }
            }
        }
        
        if (input_tensor.dtype() != torch::kBool && input_tensor.numel() > 0) {
            try {
                auto float_tensor = input_tensor.to(torch::kFloat);
                torch::resolve_conj(float_tensor);
            } catch (...) {
            }
        }
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            try {
                auto complex_tensor = input_tensor.to(torch::kComplexFloat);
                auto conj_complex = complex_tensor.conj();
                torch::resolve_conj(conj_complex);
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