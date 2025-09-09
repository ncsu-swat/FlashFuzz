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

        switch (operation_type) {
            case 0: {
                torch::std(input_tensor);
                break;
            }
            case 1: {
                if (offset >= Size) break;
                int64_t dim_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    dim_raw = 0;
                }
                int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
                torch::std(input_tensor, dim);
                break;
            }
            case 2: {
                if (offset >= Size) break;
                uint8_t keepdim_byte = Data[offset++];
                bool keepdim = keepdim_byte % 2 == 1;
                torch::std(input_tensor, c10::nullopt, 1, keepdim);
                break;
            }
            case 3: {
                if (offset + 1 >= Size) break;
                int64_t dim_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    dim_raw = 0;
                }
                int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
                uint8_t keepdim_byte = Data[offset++];
                bool keepdim = keepdim_byte % 2 == 1;
                torch::std(input_tensor, dim, 1, keepdim);
                break;
            }
            case 4: {
                if (offset >= Size) break;
                int64_t correction_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&correction_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    correction_raw = 1;
                }
                torch::std(input_tensor, c10::nullopt, correction_raw);
                break;
            }
            case 5: {
                if (offset + sizeof(int64_t) + 1 >= Size) break;
                int64_t dim_raw;
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                int64_t dim = dim_raw % (input_tensor.dim() + 2) - 1;
                
                int64_t correction_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&correction_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    correction_raw = 1;
                }
                
                uint8_t keepdim_byte = Data[offset++];
                bool keepdim = keepdim_byte % 2 == 1;
                torch::std(input_tensor, dim, correction_raw, keepdim);
                break;
            }
            case 6: {
                if (input_tensor.dim() == 0) break;
                std::vector<int64_t> dims;
                uint8_t num_dims = (Data[offset++] % input_tensor.dim()) + 1;
                for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                    int64_t dim_raw;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    } else {
                        dim_raw = i;
                    }
                    int64_t dim = dim_raw % input_tensor.dim();
                    if (dim < 0) dim += input_tensor.dim();
                    dims.push_back(dim);
                }
                torch::std(input_tensor, dims);
                break;
            }
            case 7: {
                if (input_tensor.dim() == 0 || offset + 2 >= Size) break;
                std::vector<int64_t> dims;
                uint8_t num_dims = (Data[offset++] % input_tensor.dim()) + 1;
                for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
                    int64_t dim_raw;
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    int64_t dim = dim_raw % input_tensor.dim();
                    if (dim < 0) dim += input_tensor.dim();
                    dims.push_back(dim);
                }
                
                int64_t correction_raw = 1;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&correction_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                bool keepdim = false;
                if (offset < Size) {
                    keepdim = Data[offset++] % 2 == 1;
                }
                
                torch::std(input_tensor, dims, correction_raw, keepdim);
                break;
            }
        }

        if (offset < Size) {
            auto empty_tensor = torch::empty({0});
            torch::std(empty_tensor);
            
            auto single_element = torch::tensor({42.0});
            torch::std(single_element);
            
            auto negative_correction = Data[offset++];
            torch::std(input_tensor, c10::nullopt, -static_cast<int64_t>(negative_correction));
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}