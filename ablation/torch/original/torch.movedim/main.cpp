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
        
        int input_rank = input_tensor.dim();
        if (input_rank == 0) {
            return 0;
        }
        
        uint8_t num_dims_to_move = (Data[offset++] % std::min(input_rank, 4)) + 1;
        
        if (offset + num_dims_to_move * 2 > Size) {
            return 0;
        }
        
        std::vector<int64_t> source_dims;
        std::vector<int64_t> dest_dims;
        
        for (uint8_t i = 0; i < num_dims_to_move && offset < Size; ++i) {
            int64_t source_raw;
            std::memcpy(&source_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            int64_t source_dim = source_raw % input_rank;
            if (source_dim < 0) {
                source_dim += input_rank;
            }
            
            bool already_used = false;
            for (const auto& existing : source_dims) {
                if (existing == source_dim) {
                    already_used = true;
                    break;
                }
            }
            if (!already_used) {
                source_dims.push_back(source_dim);
            }
        }
        
        for (uint8_t i = 0; i < source_dims.size() && offset < Size; ++i) {
            int64_t dest_raw;
            std::memcpy(&dest_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            int64_t dest_dim = dest_raw % input_rank;
            if (dest_dim < 0) {
                dest_dim += input_rank;
            }
            
            bool already_used = false;
            for (const auto& existing : dest_dims) {
                if (existing == dest_dim) {
                    already_used = true;
                    break;
                }
            }
            if (!already_used) {
                dest_dims.push_back(dest_dim);
            }
        }
        
        if (source_dims.empty() || dest_dims.empty()) {
            return 0;
        }
        
        size_t min_size = std::min(source_dims.size(), dest_dims.size());
        source_dims.resize(min_size);
        dest_dims.resize(min_size);
        
        if (source_dims.size() == 1) {
            torch::movedim(input_tensor, source_dims[0], dest_dims[0]);
        } else {
            torch::movedim(input_tensor, source_dims, dest_dims);
        }
        
        if (offset < Size) {
            int64_t negative_source = -static_cast<int64_t>(Data[offset] % input_rank) - 1;
            int64_t negative_dest = -static_cast<int64_t>((Data[offset] >> 4) % input_rank) - 1;
            torch::movedim(input_tensor, negative_source, negative_dest);
        }
        
        if (offset + 1 < Size) {
            int64_t large_source = static_cast<int64_t>(Data[offset]) * 1000 + input_rank;
            int64_t large_dest = static_cast<int64_t>(Data[offset + 1]) * 1000 + input_rank;
            torch::movedim(input_tensor, large_source, large_dest);
        }
        
        std::vector<int64_t> empty_source;
        std::vector<int64_t> empty_dest;
        torch::movedim(input_tensor, empty_source, empty_dest);
        
        std::vector<int64_t> duplicate_source = {0, 0};
        std::vector<int64_t> duplicate_dest = {1, 2};
        if (input_rank >= 3) {
            torch::movedim(input_tensor, duplicate_source, duplicate_dest);
        }
        
        std::vector<int64_t> mismatched_source = {0};
        std::vector<int64_t> mismatched_dest = {1, 2};
        if (input_rank >= 3) {
            torch::movedim(input_tensor, mismatched_source, mismatched_dest);
        }
        
        for (int i = 0; i < input_rank; ++i) {
            torch::movedim(input_tensor, i, (i + 1) % input_rank);
        }
        
        std::vector<int64_t> all_dims;
        std::vector<int64_t> reversed_dims;
        for (int i = 0; i < input_rank; ++i) {
            all_dims.push_back(i);
            reversed_dims.push_back(input_rank - 1 - i);
        }
        torch::movedim(input_tensor, all_dims, reversed_dims);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}