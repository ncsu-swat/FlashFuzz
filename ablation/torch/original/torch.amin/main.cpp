#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }

        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 8;

        if (operation_type == 0) {
            torch::amin(input_tensor);
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
                dim_raw = Data[offset++];
            }
            
            int64_t dim = dim_raw % (input_tensor.dim() == 0 ? 1 : input_tensor.dim());
            if (input_tensor.dim() > 0 && dim < 0) {
                dim = input_tensor.dim() + dim;
            }
            
            torch::amin(input_tensor, dim);
        }
        else if (operation_type == 2) {
            if (offset >= Size) {
                return 0;
            }
            int64_t dim_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                dim_raw = Data[offset++];
            }
            
            int64_t dim = dim_raw % (input_tensor.dim() == 0 ? 1 : input_tensor.dim());
            if (input_tensor.dim() > 0 && dim < 0) {
                dim = input_tensor.dim() + dim;
            }
            
            bool keepdim = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
            torch::amin(input_tensor, dim, keepdim);
        }
        else if (operation_type == 3) {
            if (offset >= Size) {
                return 0;
            }
            
            uint8_t num_dims_byte = Data[offset++];
            uint8_t num_dims = (num_dims_byte % 4) + 1;
            
            std::vector<int64_t> dims;
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int64_t dim_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    dim_raw = Data[offset++];
                }
                
                int64_t dim = dim_raw % (input_tensor.dim() == 0 ? 1 : input_tensor.dim());
                if (input_tensor.dim() > 0 && dim < 0) {
                    dim = input_tensor.dim() + dim;
                }
                dims.push_back(dim);
            }
            
            if (!dims.empty()) {
                torch::amin(input_tensor, dims);
            }
        }
        else if (operation_type == 4) {
            if (offset >= Size) {
                return 0;
            }
            
            uint8_t num_dims_byte = Data[offset++];
            uint8_t num_dims = (num_dims_byte % 4) + 1;
            
            std::vector<int64_t> dims;
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int64_t dim_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    dim_raw = Data[offset++];
                }
                
                int64_t dim = dim_raw % (input_tensor.dim() == 0 ? 1 : input_tensor.dim());
                if (input_tensor.dim() > 0 && dim < 0) {
                    dim = input_tensor.dim() + dim;
                }
                dims.push_back(dim);
            }
            
            bool keepdim = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
            
            if (!dims.empty()) {
                torch::amin(input_tensor, dims, keepdim);
            }
        }
        else if (operation_type == 5) {
            int64_t extreme_dim = 999999;
            torch::amin(input_tensor, extreme_dim);
        }
        else if (operation_type == 6) {
            int64_t negative_dim = -999999;
            torch::amin(input_tensor, negative_dim);
        }
        else if (operation_type == 7) {
            std::vector<int64_t> empty_dims;
            torch::amin(input_tensor, empty_dims);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}