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

        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        bool result = torch::is_conj(input);

        if (offset < Size) {
            uint8_t conj_flag = Data[offset];
            if (conj_flag % 2 == 1) {
                torch::Tensor conj_input = torch::conj(input);
                bool conj_result = torch::is_conj(conj_input);
            }
        }

        torch::Tensor cloned = input.clone();
        bool clone_result = torch::is_conj(cloned);

        torch::Tensor detached = input.detach();
        bool detach_result = torch::is_conj(detached);

        if (input.is_complex()) {
            torch::Tensor conj_tensor = torch::conj(input);
            bool conj_check = torch::is_conj(conj_tensor);
            
            torch::Tensor conj_conj = torch::conj(conj_tensor);
            bool double_conj_check = torch::is_conj(conj_conj);
        }

        if (input.numel() > 0) {
            torch::Tensor view = input.view(-1);
            bool view_result = torch::is_conj(view);
            
            if (input.dim() > 0) {
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                bool transpose_result = torch::is_conj(transposed);
            }
        }

        torch::Tensor empty_tensor = torch::empty({0}, input.options());
        bool empty_result = torch::is_conj(empty_tensor);

        torch::Tensor scalar_tensor = torch::tensor(1.0, input.options());
        bool scalar_result = torch::is_conj(scalar_tensor);

        if (input.is_complex()) {
            torch::Tensor real_part = torch::real(input);
            bool real_result = torch::is_conj(real_part);
            
            torch::Tensor imag_part = torch::imag(input);
            bool imag_result = torch::is_conj(imag_part);
        }

        if (input.numel() > 1 && input.dim() > 0) {
            torch::Tensor sliced = input.slice(0, 0, std::min(static_cast<int64_t>(2), input.size(0)));
            bool slice_result = torch::is_conj(sliced);
        }

        torch::Tensor reshaped = input.reshape({-1});
        bool reshape_result = torch::is_conj(reshaped);

        if (input.is_floating_point() || input.is_complex()) {
            torch::Tensor contiguous = input.contiguous();
            bool contiguous_result = torch::is_conj(contiguous);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}