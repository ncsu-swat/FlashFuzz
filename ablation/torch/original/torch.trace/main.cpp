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
        
        if (input_tensor.dim() == 0) {
            auto scalar_2d = input_tensor.unsqueeze(0).unsqueeze(0);
            torch::trace(scalar_2d);
        }
        
        if (input_tensor.dim() == 1) {
            auto vec_as_diag = torch::diag(input_tensor);
            torch::trace(vec_as_diag);
            
            auto vec_as_row = input_tensor.unsqueeze(0);
            auto vec_as_col = input_tensor.unsqueeze(1);
            auto outer_prod = torch::mm(vec_as_col, vec_as_row);
            torch::trace(outer_prod);
        }
        
        if (input_tensor.dim() >= 2) {
            torch::trace(input_tensor);
            
            auto first_two_dims = input_tensor.view({input_tensor.size(0), input_tensor.size(1)});
            torch::trace(first_two_dims);
            
            if (input_tensor.dim() > 2) {
                auto flattened = input_tensor.flatten(2);
                for (int64_t i = 0; i < flattened.size(2); ++i) {
                    torch::trace(flattened.select(2, i));
                }
            }
        }
        
        if (offset < Size) {
            auto second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (input_tensor.dim() >= 1 && second_tensor.dim() >= 1) {
                try {
                    auto matmul_result = torch::mm(
                        input_tensor.view({-1, input_tensor.size(-1)}),
                        second_tensor.view({second_tensor.size(0), -1})
                    );
                    torch::trace(matmul_result);
                } catch (...) {
                }
            }
        }
        
        auto zero_tensor = torch::zeros({0, 0}, input_tensor.options());
        torch::trace(zero_tensor);
        
        auto single_elem = torch::ones({1, 1}, input_tensor.options());
        torch::trace(single_elem);
        
        if (input_tensor.numel() > 0) {
            auto large_square = torch::ones({100, 100}, input_tensor.options()) * input_tensor.flatten()[0];
            torch::trace(large_square);
        }
        
        auto rect_tall = torch::randn({10, 3}, input_tensor.options());
        torch::trace(rect_tall);
        
        auto rect_wide = torch::randn({3, 10}, input_tensor.options());
        torch::trace(rect_wide);
        
        if (input_tensor.is_floating_point()) {
            auto inf_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity(), input_tensor.options());
            torch::trace(inf_tensor);
            
            auto nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), input_tensor.options());
            torch::trace(nan_tensor);
            
            auto mixed_tensor = torch::tensor({{1.0f, std::numeric_limits<float>::infinity()}, 
                                             {std::numeric_limits<float>::quiet_NaN(), -1.0f}}, input_tensor.options());
            torch::trace(mixed_tensor);
        }
        
        if (input_tensor.is_complex()) {
            auto complex_tensor = torch::complex(torch::randn({3, 3}), torch::randn({3, 3}));
            torch::trace(complex_tensor);
        }
        
        auto transposed = input_tensor.dim() >= 2 ? input_tensor.transpose(-2, -1) : input_tensor;
        if (transposed.dim() >= 2) {
            torch::trace(transposed);
        }
        
        if (input_tensor.dim() >= 2) {
            auto non_contiguous = input_tensor.slice(0, 0, input_tensor.size(0), 2);
            torch::trace(non_contiguous);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}