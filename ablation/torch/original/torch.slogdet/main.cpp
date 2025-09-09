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
        
        if (input_tensor.dim() < 2) {
            auto shape = input_tensor.sizes().vec();
            while (shape.size() < 2) {
                shape.push_back(1);
            }
            input_tensor = input_tensor.reshape(shape);
        }
        
        int64_t last_dim = input_tensor.size(-1);
        int64_t second_last_dim = input_tensor.size(-2);
        if (last_dim != second_last_dim) {
            int64_t min_dim = std::min(last_dim, second_last_dim);
            auto shape = input_tensor.sizes().vec();
            shape[shape.size()-1] = min_dim;
            shape[shape.size()-2] = min_dim;
            input_tensor = input_tensor.narrow(-1, 0, min_dim).narrow(-2, 0, min_dim);
        }
        
        auto result = torch::slogdet(input_tensor);
        auto sign = std::get<0>(result);
        auto logabsdet = std::get<1>(result);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (input_tensor2.dim() < 2) {
                auto shape2 = input_tensor2.sizes().vec();
                while (shape2.size() < 2) {
                    shape2.push_back(1);
                }
                input_tensor2 = input_tensor2.reshape(shape2);
            }
            
            int64_t last_dim2 = input_tensor2.size(-1);
            int64_t second_last_dim2 = input_tensor2.size(-2);
            if (last_dim2 != second_last_dim2) {
                int64_t min_dim2 = std::min(last_dim2, second_last_dim2);
                auto shape2 = input_tensor2.sizes().vec();
                shape2[shape2.size()-1] = min_dim2;
                shape2[shape2.size()-2] = min_dim2;
                input_tensor2 = input_tensor2.narrow(-1, 0, min_dim2).narrow(-2, 0, min_dim2);
            }
            
            auto result2 = torch::slogdet(input_tensor2);
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_result = torch::slogdet(input_tensor);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::slogdet(input_tensor);
        }
        
        auto detached_input = input_tensor.detach();
        auto detached_result = torch::slogdet(detached_input);
        
        if (input_tensor.is_cuda() == false && input_tensor.numel() > 0) {
            auto contiguous_input = input_tensor.contiguous();
            auto contiguous_result = torch::slogdet(contiguous_input);
        }
        
        auto transposed_input = input_tensor.transpose(-1, -2);
        auto transposed_result = torch::slogdet(transposed_input);
        
        if (input_tensor.size(-1) > 1 && input_tensor.size(-2) > 1) {
            auto sliced_input = input_tensor.slice(-1, 0, input_tensor.size(-1)-1).slice(-2, 0, input_tensor.size(-2)-1);
            auto sliced_result = torch::slogdet(sliced_input);
        }
        
        auto cloned_input = input_tensor.clone();
        auto cloned_result = torch::slogdet(cloned_input);
        
        if (input_tensor.dim() > 2) {
            auto squeezed_input = input_tensor;
            for (int i = 0; i < input_tensor.dim() - 2; ++i) {
                if (squeezed_input.size(i) == 1) {
                    squeezed_input = squeezed_input.squeeze(i);
                    break;
                }
            }
            if (squeezed_input.dim() >= 2) {
                auto squeezed_result = torch::slogdet(squeezed_input);
            }
        }
        
        if (input_tensor.requires_grad() == false) {
            auto grad_input = input_tensor.requires_grad_(true);
            auto grad_result = torch::slogdet(grad_input);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}