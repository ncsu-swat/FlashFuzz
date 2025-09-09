#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 6) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t flags_byte = Data[offset++];
        bool sorted = (flags_byte & 0x01) != 0;
        bool return_inverse = (flags_byte & 0x02) != 0;
        bool return_counts = (flags_byte & 0x04) != 0;
        bool use_dim = (flags_byte & 0x08) != 0;
        
        if (use_dim && offset < Size) {
            int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
            int64_t dim = static_cast<int64_t>(dim_raw);
            
            if (return_inverse && return_counts) {
                auto result = torch::unique(input_tensor, sorted, return_inverse, return_counts, dim);
                auto output = std::get<0>(result);
                auto inverse_indices = std::get<1>(result);
                auto counts = std::get<2>(result);
            } else if (return_inverse) {
                auto result = torch::unique(input_tensor, sorted, return_inverse, false, dim);
                auto output = std::get<0>(result);
                auto inverse_indices = std::get<1>(result);
            } else if (return_counts) {
                auto result = torch::unique(input_tensor, sorted, false, return_counts, dim);
                auto output = std::get<0>(result);
                auto counts = std::get<2>(result);
            } else {
                auto output = torch::unique(input_tensor, sorted, false, false, dim);
            }
        } else {
            if (return_inverse && return_counts) {
                auto result = torch::unique(input_tensor, sorted, return_inverse, return_counts);
                auto output = std::get<0>(result);
                auto inverse_indices = std::get<1>(result);
                auto counts = std::get<2>(result);
            } else if (return_inverse) {
                auto result = torch::unique(input_tensor, sorted, return_inverse, false);
                auto output = std::get<0>(result);
                auto inverse_indices = std::get<1>(result);
            } else if (return_counts) {
                auto result = torch::unique(input_tensor, sorted, false, return_counts);
                auto output = std::get<0>(result);
                auto counts = std::get<2>(result);
            } else {
                auto output = torch::unique(input_tensor, sorted, false, false);
            }
        }
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                uint8_t flags_byte2 = Data[offset++];
                bool sorted2 = (flags_byte2 & 0x01) != 0;
                bool return_inverse2 = (flags_byte2 & 0x02) != 0;
                bool return_counts2 = (flags_byte2 & 0x04) != 0;
                bool use_dim2 = (flags_byte2 & 0x08) != 0;
                
                if (use_dim2 && offset < Size) {
                    int8_t dim_raw2 = static_cast<int8_t>(Data[offset++]);
                    int64_t dim2 = static_cast<int64_t>(dim_raw2);
                    
                    auto output2 = torch::unique(input_tensor2, sorted2, return_inverse2, return_counts2, dim2);
                } else {
                    auto output2 = torch::unique(input_tensor2, sorted2, return_inverse2, return_counts2);
                }
            }
        }
        
        auto empty_tensor = torch::empty({0});
        auto empty_result = torch::unique(empty_tensor);
        
        auto scalar_tensor = torch::tensor(42.0);
        auto scalar_result = torch::unique(scalar_tensor);
        
        auto large_tensor = torch::randint(0, 10, {1000});
        auto large_result = torch::unique(large_tensor, true, true, true);
        
        auto negative_values = torch::tensor({-5, -3, -5, -1, -3});
        auto negative_result = torch::unique(negative_values);
        
        auto mixed_values = torch::tensor({0, -1, 1, 0, -1});
        auto mixed_result = torch::unique(mixed_values, false, true, true);
        
        if (input_tensor.dim() > 0) {
            for (int64_t d = -input_tensor.dim(); d < input_tensor.dim(); ++d) {
                auto dim_result = torch::unique(input_tensor, true, false, false, d);
            }
        }
        
        auto bool_tensor = torch::tensor({true, false, true, false, true});
        auto bool_result = torch::unique(bool_tensor);
        
        auto complex_tensor = torch::tensor({{1.0 + 2.0i, 3.0 + 4.0i}, {1.0 + 2.0i, 5.0 + 6.0i}});
        auto complex_result = torch::unique(complex_tensor);
        
        auto multidim_tensor = torch::randint(0, 5, {3, 4, 5});
        for (int64_t dim = 0; dim < multidim_tensor.dim(); ++dim) {
            auto multidim_result = torch::unique(multidim_tensor, true, true, true, dim);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}