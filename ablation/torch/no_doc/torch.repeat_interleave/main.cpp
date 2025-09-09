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
        uint8_t operation_type = operation_selector % 4;
        
        if (operation_type == 0) {
            if (offset >= Size) {
                return 0;
            }
            int64_t repeats_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&repeats_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                repeats_raw = 1;
            }
            
            int64_t repeats = std::abs(repeats_raw) % 100 + 1;
            
            auto result = torch::repeat_interleave(input_tensor, repeats);
        }
        else if (operation_type == 1) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t repeats_raw;
                std::memcpy(&repeats_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                int64_t repeats = std::abs(repeats_raw) % 100 + 1;
                
                if (offset < Size) {
                    int64_t dim_raw;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    } else {
                        dim_raw = 0;
                    }
                    
                    int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
                    
                    auto result = torch::repeat_interleave(input_tensor, repeats, dim);
                }
            }
        }
        else if (operation_type == 2) {
            auto repeats_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (repeats_tensor.numel() > 0 && repeats_tensor.dim() <= 1) {
                auto result = torch::repeat_interleave(input_tensor, repeats_tensor);
            }
        }
        else if (operation_type == 3) {
            auto repeats_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (repeats_tensor.numel() > 0 && repeats_tensor.dim() <= 1 && offset < Size) {
                int64_t dim_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    dim_raw = 0;
                }
                
                int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
                
                auto result = torch::repeat_interleave(input_tensor, repeats_tensor, dim);
            }
        }
        
        if (offset < Size) {
            uint8_t edge_case_selector = Data[offset++];
            uint8_t edge_case = edge_case_selector % 6;
            
            if (edge_case == 0) {
                auto result = torch::repeat_interleave(input_tensor, 0);
            }
            else if (edge_case == 1) {
                auto result = torch::repeat_interleave(input_tensor, 1000000);
            }
            else if (edge_case == 2) {
                auto empty_tensor = torch::empty({0}, input_tensor.options());
                auto result = torch::repeat_interleave(empty_tensor, 5);
            }
            else if (edge_case == 3) {
                auto negative_repeats = torch::tensor({-1, -2, -3}, torch::kInt64);
                auto result = torch::repeat_interleave(input_tensor, negative_repeats);
            }
            else if (edge_case == 4) {
                auto zero_repeats = torch::zeros({input_tensor.size(0)}, torch::kInt64);
                auto result = torch::repeat_interleave(input_tensor, zero_repeats);
            }
            else if (edge_case == 5) {
                int64_t invalid_dim = input_tensor.dim() + 10;
                auto result = torch::repeat_interleave(input_tensor, 2, invalid_dim);
            }
        }
        
        if (offset < Size) {
            auto scalar_tensor = torch::scalar_tensor(42.0, input_tensor.options());
            auto result = torch::repeat_interleave(scalar_tensor, 3);
        }
        
        if (offset < Size) {
            auto large_tensor = torch::ones({1000, 1000}, input_tensor.options());
            auto result = torch::repeat_interleave(large_tensor, 2, 0);
        }
        
        if (offset < Size && input_tensor.dim() > 0) {
            auto mismatched_repeats = torch::ones({input_tensor.size(0) + 5}, torch::kInt64);
            auto result = torch::repeat_interleave(input_tensor, mismatched_repeats);
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}