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
        
        if (!input_tensor.dtype().is_floating_point() && !input_tensor.dtype().is_complex()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        auto result = torch::geqrf(input_tensor);
        auto q_tensor = std::get<0>(result);
        auto r_tensor = std::get<1>(result);
        
        if (offset < Size) {
            auto second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (second_tensor.dim() < 2) {
                auto shape = second_tensor.sizes().vec();
                while (shape.size() < 2) {
                    shape.push_back(1);
                }
                second_tensor = second_tensor.reshape(shape);
            }
            
            if (!second_tensor.dtype().is_floating_point() && !second_tensor.dtype().is_complex()) {
                second_tensor = second_tensor.to(torch::kFloat);
            }
            
            try {
                auto result2 = torch::geqrf(second_tensor);
            } catch (...) {
            }
        }
        
        if (offset < Size && Size - offset >= 1) {
            uint8_t batch_selector = Data[offset++];
            
            if (batch_selector % 4 == 0 && input_tensor.dim() == 2) {
                auto batched_shape = input_tensor.sizes().vec();
                batched_shape.insert(batched_shape.begin(), 2);
                auto batched_tensor = input_tensor.unsqueeze(0).expand(batched_shape);
                
                try {
                    auto batched_result = torch::geqrf(batched_tensor);
                } catch (...) {
                }
            }
        }
        
        if (input_tensor.numel() > 0) {
            auto zero_tensor = torch::zeros_like(input_tensor);
            try {
                auto zero_result = torch::geqrf(zero_tensor);
            } catch (...) {
            }
            
            auto ones_tensor = torch::ones_like(input_tensor);
            try {
                auto ones_result = torch::geqrf(ones_tensor);
            } catch (...) {
            }
        }
        
        if (input_tensor.dtype().is_floating_point()) {
            auto inf_tensor = input_tensor.clone();
            if (inf_tensor.numel() > 0) {
                inf_tensor.flatten()[0] = std::numeric_limits<float>::infinity();
                try {
                    auto inf_result = torch::geqrf(inf_tensor);
                } catch (...) {
                }
            }
            
            auto nan_tensor = input_tensor.clone();
            if (nan_tensor.numel() > 0) {
                nan_tensor.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                try {
                    auto nan_result = torch::geqrf(nan_tensor);
                } catch (...) {
                }
            }
        }
        
        auto transposed = input_tensor.transpose(-2, -1);
        try {
            auto transposed_result = torch::geqrf(transposed);
        } catch (...) {
        }
        
        if (input_tensor.sizes().size() >= 2) {
            int64_t m = input_tensor.size(-2);
            int64_t n = input_tensor.size(-1);
            
            if (m != n) {
                auto square_size = std::min(m, n);
                auto shape = input_tensor.sizes().vec();
                shape[shape.size()-2] = square_size;
                shape[shape.size()-1] = square_size;
                
                try {
                    auto square_tensor = torch::zeros(shape, input_tensor.options());
                    auto square_result = torch::geqrf(square_tensor);
                } catch (...) {
                }
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