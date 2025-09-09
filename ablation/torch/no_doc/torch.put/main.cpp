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
        
        int64_t indices_numel = 1;
        for (auto dim : indices_shape) {
            indices_numel *= dim;
        }
        
        std::vector<int64_t> indices_data;
        indices_data.reserve(indices_numel);
        
        int64_t input_numel = input_tensor.numel();
        
        for (int64_t i = 0; i < indices_numel; ++i) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t raw_idx;
                std::memcpy(&raw_idx, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                if (input_numel > 0) {
                    int64_t idx = ((raw_idx % input_numel) + input_numel) % input_numel;
                    indices_data.push_back(idx);
                } else {
                    indices_data.push_back(0);
                }
            } else {
                indices_data.push_back(0);
            }
        }
        
        auto indices_tensor = torch::from_blob(indices_data.data(), indices_shape, torch::kLong).clone();
        
        auto values_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (values_tensor.numel() == 0 && indices_tensor.numel() > 0) {
            values_tensor = torch::zeros({1}, input_tensor.options());
        }
        
        if (values_tensor.numel() > 0 && indices_tensor.numel() > 0) {
            if (values_tensor.numel() == 1) {
                values_tensor = values_tensor.expand({indices_tensor.numel()});
            } else if (values_tensor.numel() != indices_tensor.numel()) {
                int64_t target_size = indices_tensor.numel();
                if (values_tensor.numel() > target_size) {
                    values_tensor = values_tensor.narrow(0, 0, target_size);
                } else {
                    int64_t repeat_factor = (target_size + values_tensor.numel() - 1) / values_tensor.numel();
                    values_tensor = values_tensor.repeat({repeat_factor});
                    if (values_tensor.numel() > target_size) {
                        values_tensor = values_tensor.narrow(0, 0, target_size);
                    }
                }
            }
        }
        
        auto result = input_tensor.put(indices_tensor, values_tensor);
        
        if (offset < Size) {
            uint8_t accumulate_flag = Data[offset++];
            bool accumulate = (accumulate_flag % 2) == 1;
            result = input_tensor.put(indices_tensor, values_tensor, accumulate);
        }
        
        auto flattened_input = input_tensor.flatten();
        if (flattened_input.numel() > 0 && indices_tensor.numel() > 0) {
            auto clamped_indices = torch::clamp(indices_tensor, 0, flattened_input.numel() - 1);
            result = flattened_input.put(clamped_indices, values_tensor);
        }
        
        if (input_tensor.numel() > 0) {
            auto negative_indices = indices_tensor - input_tensor.numel();
            result = input_tensor.put(negative_indices, values_tensor);
        }
        
        auto large_indices = indices_tensor + input_tensor.numel() * 2;
        result = input_tensor.put(large_indices, values_tensor);
        
        if (indices_tensor.numel() > 1) {
            auto duplicate_indices = torch::cat({indices_tensor, indices_tensor});
            auto duplicate_values = torch::cat({values_tensor, values_tensor});
            result = input_tensor.put(duplicate_indices, duplicate_values);
        }
        
        auto empty_indices = torch::empty({0}, torch::kLong);
        auto empty_values = torch::empty({0}, input_tensor.options());
        result = input_tensor.put(empty_indices, empty_values);
        
        if (input_tensor.dtype() != values_tensor.dtype()) {
            auto converted_values = values_tensor.to(input_tensor.dtype());
            result = input_tensor.put(indices_tensor, converted_values);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}