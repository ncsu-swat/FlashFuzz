#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 1) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        auto result = torch::resolve_conj(input_tensor);
        
        if (offset < Size) {
            uint8_t conj_flag = Data[offset];
            if (conj_flag % 2 == 1) {
                auto conj_input = torch::conj(input_tensor);
                auto conj_result = torch::resolve_conj(conj_input);
            }
        }
        
        if (input_tensor.is_complex()) {
            auto real_part = torch::real(input_tensor);
            auto imag_part = torch::imag(input_tensor);
            auto complex_tensor = torch::complex(real_part, imag_part);
            auto complex_result = torch::resolve_conj(complex_tensor);
        }
        
        if (input_tensor.numel() > 0) {
            auto view_tensor = input_tensor.view({-1});
            auto view_result = torch::resolve_conj(view_tensor);
        }
        
        if (input_tensor.dim() > 0) {
            auto transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
            auto transpose_result = torch::resolve_conj(transposed);
        }
        
        auto cloned = input_tensor.clone();
        auto clone_result = torch::resolve_conj(cloned);
        
        if (input_tensor.is_complex()) {
            auto conj_physical = torch::conj_physical(input_tensor);
            auto conj_physical_result = torch::resolve_conj(conj_physical);
        }
        
        auto detached = input_tensor.detach();
        auto detach_result = torch::resolve_conj(detached);
        
        if (input_tensor.numel() > 1 && input_tensor.dim() > 0) {
            auto sliced = input_tensor.slice(0, 0, 1);
            auto slice_result = torch::resolve_conj(sliced);
        }
        
        if (input_tensor.is_complex() && input_tensor.numel() > 0) {
            auto real_view = torch::view_as_real(input_tensor);
            auto complex_view = torch::view_as_complex(real_view);
            auto view_complex_result = torch::resolve_conj(complex_view);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}