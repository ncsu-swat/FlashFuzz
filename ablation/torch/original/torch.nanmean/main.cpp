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

        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 8;

        if (operation_type == 0) {
            auto result = torch::nanmean(input_tensor);
        }
        else if (operation_type == 1) {
            if (offset >= Size) {
                return 0;
            }
            int64_t dim_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                dim_raw = 0;
            }
            
            int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
            auto result = torch::nanmean(input_tensor, dim);
        }
        else if (operation_type == 2) {
            if (offset >= Size) {
                return 0;
            }
            bool keepdim = (Data[offset++] % 2) == 1;
            auto result = torch::nanmean(input_tensor, c10::nullopt, keepdim);
        }
        else if (operation_type == 3) {
            if (offset + sizeof(int64_t) + 1 > Size) {
                return 0;
            }
            int64_t dim_raw;
            std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            bool keepdim = (Data[offset++] % 2) == 1;
            
            int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
            auto result = torch::nanmean(input_tensor, dim, keepdim);
        }
        else if (operation_type == 4) {
            if (offset >= Size) {
                return 0;
            }
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            auto result = torch::nanmean(input_tensor, c10::nullopt, false, dtype);
        }
        else if (operation_type == 5) {
            if (offset + sizeof(int64_t) + 1 + 1 > Size) {
                return 0;
            }
            int64_t dim_raw;
            std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            bool keepdim = (Data[offset++] % 2) == 1;
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
            auto result = torch::nanmean(input_tensor, dim, keepdim, dtype);
        }
        else if (operation_type == 6) {
            if (offset + sizeof(int64_t) * 2 > Size) {
                return 0;
            }
            
            int64_t num_dims_raw;
            std::memcpy(&num_dims_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            int num_dims = std::abs(num_dims_raw) % 5;
            std::vector<int64_t> dims;
            
            for (int i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim_raw;
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
                dims.push_back(dim);
            }
            
            if (!dims.empty()) {
                auto result = torch::nanmean(input_tensor, dims);
            }
        }
        else if (operation_type == 7) {
            if (offset + sizeof(int64_t) * 2 + 2 > Size) {
                return 0;
            }
            
            int64_t num_dims_raw;
            std::memcpy(&num_dims_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            int num_dims = std::abs(num_dims_raw) % 5;
            std::vector<int64_t> dims;
            
            for (int i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim_raw;
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
                dims.push_back(dim);
            }
            
            if (offset + 2 <= Size) {
                bool keepdim = (Data[offset++] % 2) == 1;
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                if (!dims.empty()) {
                    auto result = torch::nanmean(input_tensor, dims, keepdim, dtype);
                }
            }
        }

        if (input_tensor.numel() > 0 && offset < Size) {
            auto nan_tensor = input_tensor.clone();
            if (nan_tensor.is_floating_point()) {
                auto mask = torch::rand_like(nan_tensor) < 0.3;
                nan_tensor.masked_fill_(mask, std::numeric_limits<double>::quiet_NaN());
                auto result = torch::nanmean(nan_tensor);
            }
        }

        if (input_tensor.numel() > 0 && offset < Size) {
            auto all_nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            if (all_nan_tensor.is_floating_point()) {
                auto result = torch::nanmean(all_nan_tensor);
            }
        }

        if (input_tensor.numel() == 0) {
            auto result = torch::nanmean(input_tensor);
        }

        if (input_tensor.dim() > 0) {
            int64_t invalid_dim = input_tensor.dim() + 10;
            try {
                auto result = torch::nanmean(input_tensor, invalid_dim);
            } catch (...) {
            }
        }

        if (input_tensor.dim() > 0) {
            int64_t negative_dim = -(input_tensor.dim() + 5);
            try {
                auto result = torch::nanmean(input_tensor, negative_dim);
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