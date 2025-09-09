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

        uint8_t config_byte = Data[offset++];
        
        bool use_dim = (config_byte & 0x01) != 0;
        bool keepdim = (config_byte & 0x02) != 0;
        bool use_correction = (config_byte & 0x04) != 0;
        bool use_multiple_dims = (config_byte & 0x08) != 0;
        bool use_negative_dim = (config_byte & 0x10) != 0;
        bool use_out_tensors = (config_byte & 0x20) != 0;
        
        if (!use_dim) {
            auto result = torch::var_mean(input_tensor);
            auto var_tensor = std::get<0>(result);
            auto mean_tensor = std::get<1>(result);
            return 0;
        }
        
        if (input_tensor.dim() == 0) {
            auto result = torch::var_mean(input_tensor);
            auto var_tensor = std::get<0>(result);
            auto mean_tensor = std::get<1>(result);
            return 0;
        }
        
        std::vector<int64_t> dims;
        int64_t tensor_ndim = input_tensor.dim();
        
        if (use_multiple_dims && offset + 2 < Size) {
            uint8_t num_dims = (Data[offset++] % 4) + 1;
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(Data[offset++] % (tensor_ndim * 2));
                if (use_negative_dim && dim >= tensor_ndim) {
                    dim = dim - tensor_ndim * 2;
                }
                if (dim >= tensor_ndim) {
                    dim = dim % tensor_ndim;
                }
                dims.push_back(dim);
            }
        } else if (offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset++] % (tensor_ndim * 2));
            if (use_negative_dim && dim >= tensor_ndim) {
                dim = dim - tensor_ndim * 2;
            }
            if (dim >= tensor_ndim) {
                dim = dim % tensor_ndim;
            }
            dims.push_back(dim);
        }
        
        int64_t correction = 1;
        if (use_correction && offset < Size) {
            correction = static_cast<int64_t>(Data[offset++]) - 128;
        }
        
        if (use_out_tensors) {
            try {
                auto var_out = torch::empty_like(input_tensor);
                auto mean_out = torch::empty_like(input_tensor);
                
                if (dims.empty()) {
                    auto result = torch::var_mean_out(var_out, mean_out, input_tensor, c10::nullopt, correction, keepdim);
                } else {
                    auto result = torch::var_mean_out(var_out, mean_out, input_tensor, dims, correction, keepdim);
                }
            } catch (...) {
                if (dims.empty()) {
                    auto result = torch::var_mean(input_tensor, c10::nullopt, correction, keepdim);
                } else {
                    auto result = torch::var_mean(input_tensor, dims, correction, keepdim);
                }
            }
        } else {
            if (dims.empty()) {
                auto result = torch::var_mean(input_tensor, c10::nullopt, correction, keepdim);
                auto var_tensor = std::get<0>(result);
                auto mean_tensor = std::get<1>(result);
            } else {
                auto result = torch::var_mean(input_tensor, dims, correction, keepdim);
                auto var_tensor = std::get<0>(result);
                auto mean_tensor = std::get<1>(result);
            }
        }
        
        if (offset + 4 < Size) {
            int64_t large_dim = *reinterpret_cast<const int32_t*>(Data + offset);
            offset += 4;
            try {
                auto result = torch::var_mean(input_tensor, large_dim, correction, keepdim);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            int64_t extreme_correction = *reinterpret_cast<const int64_t*>(Data + offset);
            try {
                auto result = torch::var_mean(input_tensor, dims, extreme_correction, keepdim);
            } catch (...) {
            }
        }
        
        try {
            std::vector<int64_t> invalid_dims = {tensor_ndim + 100, -tensor_ndim - 100};
            auto result = torch::var_mean(input_tensor, invalid_dims, correction, keepdim);
        } catch (...) {
        }
        
        try {
            std::vector<int64_t> duplicate_dims = {0, 0, 1, 1};
            if (tensor_ndim >= 2) {
                auto result = torch::var_mean(input_tensor, duplicate_dims, correction, keepdim);
            }
        } catch (...) {
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}