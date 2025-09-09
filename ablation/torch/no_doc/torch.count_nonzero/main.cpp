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
            torch::count_nonzero(input_tensor);
            return 0;
        }
        
        uint8_t dim_selector = Data[offset++];
        
        if (dim_selector % 3 == 0) {
            torch::count_nonzero(input_tensor);
        } else if (dim_selector % 3 == 1) {
            if (input_tensor.dim() > 0) {
                int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                torch::count_nonzero(input_tensor, dim);
            } else {
                torch::count_nonzero(input_tensor);
            }
        } else {
            if (input_tensor.dim() > 0) {
                int64_t dim = static_cast<int64_t>(dim_selector) % input_tensor.dim();
                if (dim_selector % 2 == 0) {
                    dim = -dim - 1;
                }
                torch::count_nonzero(input_tensor, dim);
            } else {
                torch::count_nonzero(input_tensor);
            }
        }
        
        if (offset < Size && input_tensor.dim() > 1) {
            std::vector<int64_t> dims;
            uint8_t num_dims = Data[offset++] % (input_tensor.dim() + 1);
            
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                if (Data[offset - 1] % 2 == 0) {
                    dim = -dim - 1;
                }
                dims.push_back(dim);
            }
            
            if (!dims.empty()) {
                torch::count_nonzero(input_tensor, dims);
            }
        }
        
        if (offset < Size) {
            auto cloned_tensor = input_tensor.clone();
            cloned_tensor = cloned_tensor.to(torch::kFloat);
            torch::count_nonzero(cloned_tensor);
            
            if (cloned_tensor.dim() > 0) {
                torch::count_nonzero(cloned_tensor, 0);
            }
        }
        
        if (offset < Size && input_tensor.numel() > 0) {
            auto reshaped = input_tensor.view({-1});
            torch::count_nonzero(reshaped);
            
            if (input_tensor.dim() > 1) {
                auto flattened = input_tensor.flatten();
                torch::count_nonzero(flattened);
            }
        }
        
        if (offset < Size) {
            auto empty_tensor = torch::empty({0});
            torch::count_nonzero(empty_tensor);
            
            auto zero_tensor = torch::zeros_like(input_tensor);
            torch::count_nonzero(zero_tensor);
            
            auto ones_tensor = torch::ones_like(input_tensor);
            torch::count_nonzero(ones_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}