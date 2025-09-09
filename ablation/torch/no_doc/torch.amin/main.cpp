#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::amin(input_tensor);
            return 0;
        }
        
        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 4;
        
        if (operation_type == 0) {
            torch::amin(input_tensor);
        }
        else if (operation_type == 1) {
            if (offset >= Size) {
                torch::amin(input_tensor);
                return 0;
            }
            
            uint8_t dim_selector = Data[offset++];
            int64_t tensor_rank = input_tensor.dim();
            
            if (tensor_rank == 0) {
                torch::amin(input_tensor);
            } else {
                int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_selector)) % (tensor_rank * 2 + 1) - tensor_rank;
                torch::amin(input_tensor, dim);
            }
        }
        else if (operation_type == 2) {
            if (offset >= Size) {
                torch::amin(input_tensor);
                return 0;
            }
            
            uint8_t dim_selector = Data[offset++];
            int64_t tensor_rank = input_tensor.dim();
            
            if (tensor_rank == 0) {
                torch::amin(input_tensor);
            } else {
                int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_selector)) % (tensor_rank * 2 + 1) - tensor_rank;
                
                if (offset >= Size) {
                    torch::amin(input_tensor, dim);
                    return 0;
                }
                
                uint8_t keepdim_selector = Data[offset++];
                bool keepdim = (keepdim_selector % 2) == 1;
                torch::amin(input_tensor, dim, keepdim);
            }
        }
        else {
            if (offset + 1 >= Size) {
                torch::amin(input_tensor);
                return 0;
            }
            
            uint8_t num_dims_selector = Data[offset++];
            int64_t tensor_rank = input_tensor.dim();
            
            if (tensor_rank == 0) {
                torch::amin(input_tensor);
                return 0;
            }
            
            uint8_t num_dims = (num_dims_selector % tensor_rank) + 1;
            std::vector<int64_t> dims;
            
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                uint8_t dim_selector = Data[offset++];
                int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_selector)) % (tensor_rank * 2 + 1) - tensor_rank;
                dims.push_back(dim);
            }
            
            if (dims.empty()) {
                torch::amin(input_tensor);
            } else {
                if (offset < Size) {
                    uint8_t keepdim_selector = Data[offset++];
                    bool keepdim = (keepdim_selector % 2) == 1;
                    torch::amin(input_tensor, dims, keepdim);
                } else {
                    torch::amin(input_tensor, dims);
                }
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