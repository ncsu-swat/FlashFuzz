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
        
        uint8_t indices_rank_byte = Data[offset++];
        uint8_t indices_rank = fuzzer_utils::parseRank(indices_rank_byte);
        
        if (indices_rank == 0) {
            indices_rank = 1;
        }
        
        auto indices_shape = fuzzer_utils::parseShape(Data, offset, Size, indices_rank);
        
        if (offset >= Size) {
            return 0;
        }
        
        int64_t indices_numel = 1;
        for (auto dim : indices_shape) {
            indices_numel *= dim;
        }
        
        std::vector<int64_t> indices_data;
        indices_data.reserve(indices_numel);
        
        for (int64_t i = 0; i < indices_numel && offset < Size; ++i) {
            int64_t idx_raw = 0;
            size_t bytes_to_read = std::min(sizeof(int64_t), Size - offset);
            std::memcpy(&idx_raw, Data + offset, bytes_to_read);
            offset += bytes_to_read;
            indices_data.push_back(idx_raw);
        }
        
        while (indices_data.size() < static_cast<size_t>(indices_numel)) {
            indices_data.push_back(0);
        }
        
        auto indices_tensor = torch::from_blob(indices_data.data(), indices_shape, torch::kLong).clone();
        
        if (offset >= Size) {
            return 0;
        }
        
        auto values_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset < Size) {
            bool accumulate = (Data[offset] % 2) == 1;
            
            input_tensor.put_(indices_tensor, values_tensor, accumulate);
        } else {
            input_tensor.put_(indices_tensor, values_tensor);
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_indices = torch::empty({0}, torch::kLong);
            auto empty_values = torch::empty({0}, input_tensor.dtype());
            input_tensor.put_(empty_indices, empty_values);
        }
        
        if (indices_tensor.numel() > 0) {
            auto large_indices = indices_tensor * 1000000;
            try {
                input_tensor.put_(large_indices, values_tensor);
            } catch (...) {
            }
        }
        
        if (indices_tensor.numel() > 0) {
            auto negative_indices = -torch::abs(indices_tensor) - 1;
            try {
                input_tensor.put_(negative_indices, values_tensor);
            } catch (...) {
            }
        }
        
        if (input_tensor.dim() > 0 && values_tensor.numel() > 0) {
            auto broadcast_values = values_tensor.expand({indices_tensor.numel()});
            try {
                input_tensor.put_(indices_tensor, broadcast_values);
            } catch (...) {
            }
        }
        
        if (indices_tensor.numel() > 1) {
            auto duplicate_indices = torch::cat({indices_tensor.flatten(), indices_tensor.flatten()});
            auto duplicate_values = torch::cat({values_tensor.flatten(), values_tensor.flatten()});
            try {
                input_tensor.put_(duplicate_indices, duplicate_values, true);
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