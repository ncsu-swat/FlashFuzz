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
        
        uint8_t source_type = Data[offset++];
        if (offset >= Size) {
            return 0;
        }
        
        if (source_type % 2 == 0) {
            int64_t source_dim = static_cast<int64_t>(Data[offset++]) - 128;
            if (offset >= Size) {
                return 0;
            }
            int64_t dest_dim = static_cast<int64_t>(Data[offset++]) - 128;
            
            torch::moveaxis(input_tensor, source_dim, dest_dim);
        } else {
            uint8_t num_axes = (Data[offset++] % 4) + 1;
            if (offset + 2 * num_axes > Size) {
                return 0;
            }
            
            std::vector<int64_t> source_dims;
            std::vector<int64_t> dest_dims;
            
            for (uint8_t i = 0; i < num_axes; ++i) {
                source_dims.push_back(static_cast<int64_t>(Data[offset++]) - 128);
            }
            
            for (uint8_t i = 0; i < num_axes; ++i) {
                dest_dims.push_back(static_cast<int64_t>(Data[offset++]) - 128);
            }
            
            torch::moveaxis(input_tensor, source_dims, dest_dims);
        }
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor2.dim() > 0) {
                int64_t extreme_source = std::numeric_limits<int64_t>::max();
                int64_t extreme_dest = std::numeric_limits<int64_t>::min();
                torch::moveaxis(input_tensor2, extreme_source, extreme_dest);
            }
        }
        
        if (offset < Size) {
            auto input_tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor3.dim() > 1) {
                std::vector<int64_t> large_source_dims;
                std::vector<int64_t> large_dest_dims;
                
                for (int i = 0; i < 100; ++i) {
                    large_source_dims.push_back(i);
                    large_dest_dims.push_back(-i);
                }
                
                torch::moveaxis(input_tensor3, large_source_dims, large_dest_dims);
            }
        }
        
        if (offset < Size) {
            auto input_tensor4 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor4.dim() > 0) {
                std::vector<int64_t> empty_source;
                std::vector<int64_t> empty_dest;
                torch::moveaxis(input_tensor4, empty_source, empty_dest);
            }
        }
        
        if (offset < Size) {
            auto input_tensor5 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor5.dim() > 1) {
                std::vector<int64_t> mismatched_source = {0};
                std::vector<int64_t> mismatched_dest = {1, 2};
                torch::moveaxis(input_tensor5, mismatched_source, mismatched_dest);
            }
        }
        
        if (offset < Size) {
            auto input_tensor6 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor6.dim() > 2) {
                std::vector<int64_t> duplicate_source = {0, 0, 1};
                std::vector<int64_t> duplicate_dest = {1, 2, 0};
                torch::moveaxis(input_tensor6, duplicate_source, duplicate_dest);
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