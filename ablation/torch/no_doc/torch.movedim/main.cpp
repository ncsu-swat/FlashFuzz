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
        
        torch::movedim(input_tensor, source_dim, dest_dim);
        
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
                torch::movedim(input_tensor, source_dims, dest_dims);
            }
        }
        
        if (offset < Size) {
            uint8_t extreme_count = Data[offset++] % 3;
            for (uint8_t i = 0; i < extreme_count && offset + 7 < Size; ++i) {
                int64_t extreme_source, extreme_dest;
                std::memcpy(&extreme_source, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                std::memcpy(&extreme_dest, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                torch::movedim(input_tensor, extreme_source, extreme_dest);
            }
        }
        
        if (offset < Size) {
            std::vector<int64_t> large_source_dims;
            std::vector<int64_t> large_dest_dims;
            
            uint8_t large_count = Data[offset++] % 10;
            for (uint8_t i = 0; i < large_count && offset + 15 < Size; ++i) {
                int64_t src, dst;
                std::memcpy(&src, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                std::memcpy(&dst, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                large_source_dims.push_back(src);
                large_dest_dims.push_back(dst);
            }
            
            if (!large_source_dims.empty() && !large_dest_dims.empty()) {
                torch::movedim(input_tensor, large_source_dims, large_dest_dims);
            }
        }
        
        if (offset < Size) {
            std::vector<int64_t> duplicate_sources;
            std::vector<int64_t> duplicate_dests;
            
            uint8_t dup_count = Data[offset++] % 4;
            if (dup_count > 0 && offset < Size) {
                int64_t base_dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
                for (uint8_t i = 0; i < dup_count; ++i) {
                    duplicate_sources.push_back(base_dim);
                    duplicate_dests.push_back(base_dim + i);
                }
                torch::movedim(input_tensor, duplicate_sources, duplicate_dests);
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