#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        auto input = fuzzer_utils::createTensor(Data, Size, offset);
        auto mat2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input.dim() != 2 || mat2.dim() != 2) {
            input = input.view({-1, 1});
            mat2 = mat2.view({1, -1});
            
            if (input.numel() == 0) {
                input = torch::ones({1, 1}, input.options());
            }
            if (mat2.numel() == 0) {
                mat2 = torch::ones({1, 1}, mat2.options());
            }
        }
        
        auto input_sizes = input.sizes();
        auto mat2_sizes = mat2.sizes();
        
        if (input_sizes.size() >= 2 && mat2_sizes.size() >= 2) {
            int64_t input_cols = input_sizes[input_sizes.size() - 1];
            int64_t mat2_rows = mat2_sizes[mat2_sizes.size() - 2];
            
            if (input_cols != mat2_rows) {
                if (input_cols == 0 || mat2_rows == 0) {
                    input = torch::ones({1, 1}, input.options());
                    mat2 = torch::ones({1, 1}, mat2.options());
                } else {
                    int64_t common_dim = std::min(input_cols, mat2_rows);
                    if (common_dim <= 0) common_dim = 1;
                    
                    input = input.narrow(-1, 0, std::min(input_cols, common_dim));
                    mat2 = mat2.narrow(-2, 0, std::min(mat2_rows, common_dim));
                    
                    if (input.size(-1) != common_dim) {
                        auto padding = torch::zeros({input.size(0), common_dim - input.size(-1)}, input.options());
                        input = torch::cat({input, padding}, -1);
                    }
                    if (mat2.size(-2) != common_dim) {
                        auto padding = torch::zeros({common_dim - mat2.size(-2), mat2.size(-1)}, mat2.options());
                        mat2 = torch::cat({padding, mat2}, -2);
                    }
                }
            }
        }
        
        auto result = torch::mm(input, mat2);
        
        if (offset < Size) {
            torch::Tensor out_tensor;
            try {
                out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (out_tensor.dim() == 2) {
                    auto expected_shape = std::vector<int64_t>{input.size(0), mat2.size(1)};
                    if (out_tensor.size(0) == expected_shape[0] && out_tensor.size(1) == expected_shape[1]) {
                        torch::mm_out(out_tensor, input, mat2);
                    }
                }
            } catch (...) {
            }
        }
        
        if (input.dtype() != mat2.dtype()) {
            try {
                auto input_converted = input.to(mat2.dtype());
                auto result2 = torch::mm(input_converted, mat2);
            } catch (...) {
            }
            
            try {
                auto mat2_converted = mat2.to(input.dtype());
                auto result3 = torch::mm(input, mat2_converted);
            } catch (...) {
            }
        }
        
        if (input.is_floating_point() && mat2.is_floating_point()) {
            try {
                auto input_sparse = input.to_sparse();
                auto result_sparse = torch::mm(input_sparse, mat2);
            } catch (...) {
            }
            
            try {
                auto mat2_sparse = mat2.to_sparse();
                auto result_sparse2 = torch::mm(input, mat2_sparse);
            } catch (...) {
            }
        }
        
        auto input_t = input.t();
        auto mat2_t = mat2.t();
        try {
            if (input_t.size(1) == mat2_t.size(0)) {
                auto result_transposed = torch::mm(input_t, mat2_t);
            }
        } catch (...) {
        }
        
        if (input.numel() > 0 && mat2.numel() > 0) {
            try {
                auto input_noncontig = input.transpose(0, 1).transpose(0, 1);
                auto result_noncontig = torch::mm(input_noncontig, mat2);
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