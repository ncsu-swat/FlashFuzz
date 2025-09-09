#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t k_byte = Data[offset++];
        int64_t k = static_cast<int64_t>(k_byte) + 1;
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t dim_byte = Data[offset++];
        int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_byte));
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t largest_byte = Data[offset++];
        bool largest = (largest_byte % 2) == 0;
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t sorted_byte = Data[offset++];
        bool sorted = (sorted_byte % 2) == 0;
        
        torch::topk(input_tensor, k);
        
        torch::topk(input_tensor, k, dim);
        
        torch::topk(input_tensor, k, dim, largest);
        
        torch::topk(input_tensor, k, dim, largest, sorted);
        
        if (input_tensor.numel() > 0) {
            torch::topk(input_tensor, 1);
        }
        
        if (input_tensor.dim() > 0) {
            int64_t last_dim = input_tensor.dim() - 1;
            int64_t last_dim_size = input_tensor.size(last_dim);
            if (last_dim_size > 0) {
                int64_t safe_k = std::min(k, last_dim_size);
                torch::topk(input_tensor, safe_k, last_dim);
            }
        }
        
        torch::topk(input_tensor, k, -1);
        torch::topk(input_tensor, k, -2);
        
        if (input_tensor.dim() >= 2) {
            torch::topk(input_tensor, k, 0);
            torch::topk(input_tensor, k, 1);
        }
        
        int64_t large_k = k * 1000;
        torch::topk(input_tensor, large_k);
        
        int64_t zero_k = 0;
        torch::topk(input_tensor, zero_k);
        
        int64_t negative_k = -k;
        torch::topk(input_tensor, negative_k);
        
        if (input_tensor.dim() > 0) {
            int64_t invalid_dim = input_tensor.dim() + 10;
            torch::topk(input_tensor, k, invalid_dim);
        }
        
        int64_t very_negative_dim = -1000;
        torch::topk(input_tensor, k, very_negative_dim);
        
        auto empty_tensor = torch::empty({0});
        torch::topk(empty_tensor, k);
        
        auto scalar_tensor = torch::tensor(42.0);
        torch::topk(scalar_tensor, k);
        
        auto large_tensor = torch::randn({1000, 1000});
        torch::topk(large_tensor, k);
        
        if (offset < Size) {
            uint8_t extra_byte = Data[offset++];
            int64_t extra_k = static_cast<int64_t>(extra_byte);
            torch::topk(input_tensor, extra_k, dim, !largest, !sorted);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}