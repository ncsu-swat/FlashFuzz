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
        bool unbiased = (config_byte & 0x01) != 0;
        bool keepdim = (config_byte & 0x02) != 0;
        bool use_dim = (config_byte & 0x04) != 0;
        bool use_correction = (config_byte & 0x08) != 0;

        if (input_tensor.numel() == 0) {
            auto result = torch::var_mean(input_tensor, unbiased);
            return 0;
        }

        if (!use_dim) {
            auto result = torch::var_mean(input_tensor, unbiased);
            std::get<0>(result);
            std::get<1>(result);
            
            if (use_correction && offset < Size) {
                int64_t correction_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&correction_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    correction_raw = 0;
                }
                auto result_corr = torch::var_mean(input_tensor, correction_raw);
                std::get<0>(result_corr);
                std::get<1>(result_corr);
            }
        } else {
            if (offset >= Size) {
                return 0;
            }
            
            int rank = input_tensor.dim();
            if (rank == 0) {
                auto result = torch::var_mean(input_tensor, unbiased);
                return 0;
            }
            
            uint8_t dim_config = Data[offset++];
            int64_t dim = static_cast<int64_t>(dim_config) % (2 * rank) - rank;
            
            auto result = torch::var_mean(input_tensor, dim, unbiased, keepdim);
            std::get<0>(result);
            std::get<1>(result);
            
            if (use_correction && offset < Size) {
                int64_t correction_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&correction_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    correction_raw = 0;
                }
                auto result_corr = torch::var_mean(input_tensor, dim, correction_raw, keepdim);
                std::get<0>(result_corr);
                std::get<1>(result_corr);
            }
            
            if (offset < Size && rank > 1) {
                std::vector<int64_t> dims;
                uint8_t num_dims = (Data[offset++] % rank) + 1;
                
                for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                    uint8_t dim_byte = Data[offset++];
                    int64_t multi_dim = static_cast<int64_t>(dim_byte) % (2 * rank) - rank;
                    dims.push_back(multi_dim);
                }
                
                if (!dims.empty()) {
                    auto result_multi = torch::var_mean(input_tensor, dims, unbiased, keepdim);
                    std::get<0>(result_multi);
                    std::get<1>(result_multi);
                    
                    if (use_correction && offset < Size) {
                        int64_t correction_raw;
                        if (offset + sizeof(int64_t) <= Size) {
                            std::memcpy(&correction_raw, Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                        } else {
                            correction_raw = 0;
                        }
                        auto result_multi_corr = torch::var_mean(input_tensor, dims, correction_raw, keepdim);
                        std::get<0>(result_multi_corr);
                        std::get<1>(result_multi_corr);
                    }
                }
            }
        }

        if (offset < Size) {
            auto second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (second_tensor.sizes() == input_tensor.sizes()) {
                auto combined = input_tensor + second_tensor;
                auto result_combined = torch::var_mean(combined, unbiased);
                std::get<0>(result_combined);
                std::get<1>(result_combined);
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