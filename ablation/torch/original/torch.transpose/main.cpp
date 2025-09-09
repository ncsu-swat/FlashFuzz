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
        
        int64_t tensor_rank = input_tensor.dim();
        
        if (tensor_rank == 0) {
            return 0;
        }
        
        uint8_t dim0_byte = Data[offset % Size];
        offset++;
        uint8_t dim1_byte = Data[offset % Size];
        offset++;
        
        int64_t dim0_raw = static_cast<int64_t>(static_cast<int8_t>(dim0_byte));
        int64_t dim1_raw = static_cast<int64_t>(static_cast<int8_t>(dim1_byte));
        
        int64_t dim0 = dim0_raw % (tensor_rank * 2) - tensor_rank;
        int64_t dim1 = dim1_raw % (tensor_rank * 2) - tensor_rank;
        
        auto result = torch::transpose(input_tensor, dim0, dim1);
        
        if (offset < Size) {
            uint8_t extra_byte = Data[offset % Size];
            int64_t large_dim0 = static_cast<int64_t>(extra_byte) * 1000;
            int64_t large_dim1 = -large_dim0;
            
            try {
                auto result2 = torch::transpose(input_tensor, large_dim0, large_dim1);
            } catch (...) {
            }
        }
        
        if (tensor_rank >= 2) {
            auto result_identity = torch::transpose(input_tensor, 0, 0);
            auto result_chain = torch::transpose(torch::transpose(input_tensor, 0, 1), 0, 1);
        }
        
        if (input_tensor.numel() > 0) {
            try {
                auto result_neg = torch::transpose(input_tensor, -1, -2);
            } catch (...) {
            }
        }
        
        auto empty_tensor = torch::empty({0, 5, 3});
        try {
            auto empty_result = torch::transpose(empty_tensor, 0, 1);
        } catch (...) {
        }
        
        auto scalar_tensor = torch::tensor(42.0);
        try {
            auto scalar_result = torch::transpose(scalar_tensor, 0, 0);
        } catch (...) {
        }
        
        if (tensor_rank >= 3) {
            for (int64_t i = 0; i < tensor_rank; i++) {
                for (int64_t j = 0; j < tensor_rank; j++) {
                    try {
                        auto perm_result = torch::transpose(input_tensor, i, j);
                    } catch (...) {
                    }
                }
            }
        }
        
        auto large_tensor = torch::zeros({2, 3, 4, 5});
        try {
            auto large_result = torch::transpose(large_tensor, 1000, -1000);
        } catch (...) {
        }
        
        if (input_tensor.is_sparse()) {
            try {
                auto sparse_result = torch::transpose(input_tensor, 0, 1);
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