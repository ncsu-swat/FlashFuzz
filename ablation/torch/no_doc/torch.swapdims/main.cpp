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
        int64_t dim0 = static_cast<int64_t>(static_cast<int8_t>(dim0_byte));
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t dim1_byte = Data[offset++];
        int64_t dim1 = static_cast<int64_t>(static_cast<int8_t>(dim1_byte));
        
        auto result = torch::swapdims(tensor, dim0, dim1);
        
        if (tensor.dim() >= 2) {
            auto result2 = torch::swapdims(tensor, -1, 0);
            auto result3 = torch::swapdims(tensor, tensor.dim() - 1, 0);
        }
        
        if (tensor.dim() >= 3) {
            auto result4 = torch::swapdims(tensor, -2, -1);
            auto result5 = torch::swapdims(tensor, 1, -1);
        }
        
        if (tensor.dim() == 0) {
            auto result_scalar = torch::swapdims(tensor, 0, 0);
        }
        
        auto large_dim = tensor.dim() + 100;
        auto result_large = torch::swapdims(tensor, 0, large_dim);
        
        auto negative_large = -large_dim;
        auto result_neg_large = torch::swapdims(tensor, negative_large, 0);
        
        auto result_same = torch::swapdims(tensor, 0, 0);
        
        if (tensor.dim() >= 1) {
            auto result_boundary = torch::swapdims(tensor, tensor.dim() - 1, -(tensor.dim()));
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}