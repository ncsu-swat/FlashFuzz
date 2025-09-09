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
        
        int64_t tensor_rank = input_tensor.dim();
        if (tensor_rank == 0) {
            return 0;
        }
        
        uint8_t source_byte = Data[offset++];
        uint8_t dest_byte = Data[offset++];
        
        int64_t source_dim = static_cast<int64_t>(static_cast<int8_t>(source_byte));
        int64_t dest_dim = static_cast<int64_t>(static_cast<int8_t>(dest_byte));
        
        torch::moveaxis(input_tensor, source_dim, dest_dim);
        
        if (offset < Size) {
            uint8_t multi_source_count = Data[offset++] % 5;
            uint8_t multi_dest_count = Data[offset++] % 5;
            
            std::vector<int64_t> source_dims;
            std::vector<int64_t> dest_dims;
            
            for (uint8_t i = 0; i < multi_source_count && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                source_dims.push_back(dim);
            }
            
            for (uint8_t i = 0; i < multi_dest_count && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                dest_dims.push_back(dim);
            }
            
            if (!source_dims.empty() && !dest_dims.empty()) {
                torch::moveaxis(input_tensor, source_dims, dest_dims);
            }
        }
        
        if (offset < Size) {
            auto second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (second_tensor.dim() > 0 && offset < Size) {
                int64_t second_rank = second_tensor.dim();
                uint8_t src_byte = Data[offset++];
                uint8_t dst_byte = Data[offset++];
                
                int64_t src = static_cast<int64_t>(static_cast<int8_t>(src_byte)) % (second_rank * 2 + 1) - second_rank;
                int64_t dst = static_cast<int64_t>(static_cast<int8_t>(dst_byte)) % (second_rank * 2 + 1) - second_rank;
                
                torch::moveaxis(second_tensor, src, dst);
            }
        }
        
        if (offset < Size && input_tensor.dim() >= 2) {
            std::vector<int64_t> large_source_dims;
            std::vector<int64_t> large_dest_dims;
            
            uint8_t num_moves = Data[offset++] % 10;
            for (uint8_t i = 0; i < num_moves && offset + 1 < Size; ++i) {
                int64_t src = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                int64_t dst = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                large_source_dims.push_back(src);
                large_dest_dims.push_back(dst);
            }
            
            if (!large_source_dims.empty() && large_source_dims.size() == large_dest_dims.size()) {
                torch::moveaxis(input_tensor, large_source_dims, large_dest_dims);
            }
        }
        
        if (offset < Size && input_tensor.dim() > 0) {
            int64_t extreme_source = std::numeric_limits<int64_t>::max();
            int64_t extreme_dest = std::numeric_limits<int64_t>::min();
            torch::moveaxis(input_tensor, extreme_source, extreme_dest);
        }
        
        if (offset < Size && input_tensor.dim() > 0) {
            std::vector<int64_t> duplicate_sources = {0, 0, 0};
            std::vector<int64_t> duplicate_dests = {1, 2, 3};
            torch::moveaxis(input_tensor, duplicate_sources, duplicate_dests);
        }
        
        if (offset < Size && input_tensor.dim() > 1) {
            std::vector<int64_t> mismatched_sources = {0, 1};
            std::vector<int64_t> mismatched_dests = {2};
            torch::moveaxis(input_tensor, mismatched_sources, mismatched_dests);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}