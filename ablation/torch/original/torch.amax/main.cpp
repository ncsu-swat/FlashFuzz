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
            auto result = torch::amax(input_tensor);
        }
        else if (operation_type == 1) {
            if (offset >= Size) return 0;
            int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
            int64_t dim = static_cast<int64_t>(dim_raw);
            auto result = torch::amax(input_tensor, dim);
        }
        else if (operation_type == 2) {
            if (offset >= Size) return 0;
            int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
            int64_t dim = static_cast<int64_t>(dim_raw);
            bool keepdim = (Data[offset % Size] % 2) == 1;
            auto result = torch::amax(input_tensor, dim, keepdim);
        }
        else if (operation_type == 3) {
            if (offset + 1 >= Size) return 0;
            uint8_t num_dims = (Data[offset++] % 4) + 1;
            std::vector<int64_t> dims;
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
                dims.push_back(static_cast<int64_t>(dim_raw));
            }
            if (!dims.empty()) {
                auto result = torch::amax(input_tensor, dims);
            }
        }
        else if (operation_type == 4) {
            if (offset + 1 >= Size) return 0;
            uint8_t num_dims = (Data[offset++] % 4) + 1;
            std::vector<int64_t> dims;
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
                dims.push_back(static_cast<int64_t>(dim_raw));
            }
            bool keepdim = (Data[offset % Size] % 2) == 1;
            if (!dims.empty()) {
                auto result = torch::amax(input_tensor, dims, keepdim);
            }
        }
        else if (operation_type == 5) {
            if (offset >= Size) return 0;
            int64_t negative_dim = -static_cast<int64_t>((Data[offset++] % 10) + 1);
            auto result = torch::amax(input_tensor, negative_dim);
        }
        else if (operation_type == 6) {
            if (offset >= Size) return 0;
            int64_t large_dim = static_cast<int64_t>(Data[offset++]) + 1000;
            auto result = torch::amax(input_tensor, large_dim);
        }
        else if (operation_type == 7) {
            std::vector<int64_t> empty_dims;
            auto result = torch::amax(input_tensor, empty_dims);
        }
        
        if (offset < Size) {
            uint8_t out_test = Data[offset++] % 3;
            if (out_test == 0) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (offset < Size) {
                    int8_t dim_raw = static_cast<int8_t>(Data[offset % Size]);
                    int64_t dim = static_cast<int64_t>(dim_raw);
                    torch::amax_out(out_tensor, input_tensor, dim);
                }
            }
            else if (out_test == 1) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (offset < Size) {
                    int8_t dim_raw = static_cast<int8_t>(Data[offset % Size]);
                    int64_t dim = static_cast<int64_t>(dim_raw);
                    bool keepdim = ((offset + 1 < Size ? Data[offset + 1] : 0) % 2) == 1;
                    torch::amax_out(out_tensor, input_tensor, dim, keepdim);
                }
            }
            else if (out_test == 2) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::amax_out(out_tensor, input_tensor);
            }
        }
        
        if (input_tensor.numel() == 0) {
            auto result = torch::amax(input_tensor);
        }
        
        if (input_tensor.dim() > 0) {
            int64_t last_dim = input_tensor.dim() - 1;
            auto result = torch::amax(input_tensor, last_dim);
            
            int64_t first_dim = 0;
            auto result2 = torch::amax(input_tensor, first_dim, true);
        }
        
        if (input_tensor.dim() >= 2) {
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < input_tensor.dim(); ++i) {
                all_dims.push_back(i);
            }
            auto result = torch::amax(input_tensor, all_dims);
        }
        
        if (input_tensor.dim() >= 3) {
            std::vector<int64_t> partial_dims = {0, 2};
            auto result = torch::amax(input_tensor, partial_dims, false);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}