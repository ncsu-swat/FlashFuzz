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
        
        auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t dim0_byte = Data[offset++];
        int64_t dim0 = static_cast<int64_t>(dim0_byte) - 128;
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t dim1_byte = Data[offset++];
        int64_t dim1 = static_cast<int64_t>(dim1_byte) - 128;
        
        auto result = torch::transpose(tensor, dim0, dim1);
        
        if (tensor.dim() >= 2) {
            auto result2 = tensor.transpose(dim0, dim1);
        }
        
        if (tensor.dim() >= 1) {
            auto result3 = torch::transpose(tensor, 0, -1);
            auto result4 = torch::transpose(tensor, -1, 0);
        }
        
        if (tensor.dim() >= 3) {
            auto result5 = torch::transpose(tensor, 1, 2);
            auto result6 = torch::transpose(tensor, -2, -1);
        }
        
        auto result7 = torch::transpose(tensor, 0, 0);
        
        if (tensor.dim() > 0) {
            int64_t last_dim = tensor.dim() - 1;
            auto result8 = torch::transpose(tensor, 0, last_dim);
            auto result9 = torch::transpose(tensor, last_dim, 0);
        }
        
        int64_t very_large_dim = 1000000;
        auto result10 = torch::transpose(tensor, 0, very_large_dim);
        
        int64_t very_negative_dim = -1000000;
        auto result11 = torch::transpose(tensor, very_negative_dim, 0);
        
        auto result12 = torch::transpose(tensor, very_negative_dim, very_large_dim);
        
        if (tensor.numel() > 0) {
            auto result13 = torch::transpose(tensor.view({-1}), 0, 0);
        }
        
        if (tensor.dim() >= 2) {
            auto reshaped = tensor.view({-1, tensor.size(-1)});
            auto result14 = torch::transpose(reshaped, 0, 1);
        }
        
        auto empty_tensor = torch::empty({0, 5, 3});
        auto result15 = torch::transpose(empty_tensor, 0, 1);
        auto result16 = torch::transpose(empty_tensor, 1, 2);
        
        auto scalar_tensor = torch::scalar_tensor(42.0);
        auto result17 = torch::transpose(scalar_tensor, 0, 0);
        
        auto one_d_tensor = torch::ones({10});
        auto result18 = torch::transpose(one_d_tensor, 0, 0);
        
        if (tensor.dim() >= 4) {
            for (int i = 0; i < tensor.dim(); ++i) {
                for (int j = 0; j < tensor.dim(); ++j) {
                    auto result_ij = torch::transpose(tensor, i, j);
                }
            }
        }
        
        if (tensor.is_floating_point()) {
            auto nan_tensor = torch::full_like(tensor, std::numeric_limits<float>::quiet_NaN());
            auto result19 = torch::transpose(nan_tensor, 0, tensor.dim() > 1 ? 1 : 0);
            
            auto inf_tensor = torch::full_like(tensor, std::numeric_limits<float>::infinity());
            auto result20 = torch::transpose(inf_tensor, 0, tensor.dim() > 1 ? 1 : 0);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}