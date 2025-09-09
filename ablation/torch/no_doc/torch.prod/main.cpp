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
        
        auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::prod(tensor);
            return 0;
        }
        
        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 4;
        
        if (operation_type == 0) {
            torch::prod(tensor);
        }
        else if (operation_type == 1) {
            if (offset >= Size) {
                torch::prod(tensor);
                return 0;
            }
            
            uint8_t dim_selector = Data[offset++];
            int64_t tensor_rank = tensor.dim();
            
            if (tensor_rank == 0) {
                torch::prod(tensor);
            } else {
                int64_t dim_range = tensor_rank * 2;
                int64_t dim = (dim_selector % dim_range) - tensor_rank;
                torch::prod(tensor, dim);
            }
        }
        else if (operation_type == 2) {
            if (offset >= Size) {
                torch::prod(tensor);
                return 0;
            }
            
            uint8_t dim_selector = Data[offset++];
            int64_t tensor_rank = tensor.dim();
            
            if (tensor_rank == 0) {
                torch::prod(tensor);
            } else {
                int64_t dim_range = tensor_rank * 2;
                int64_t dim = (dim_selector % dim_range) - tensor_rank;
                
                uint8_t keepdim_selector = 0;
                if (offset < Size) {
                    keepdim_selector = Data[offset++];
                }
                bool keepdim = (keepdim_selector % 2) == 1;
                
                torch::prod(tensor, dim, keepdim);
            }
        }
        else {
            if (offset >= Size) {
                torch::prod(tensor);
                return 0;
            }
            
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType out_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            if (offset < Size) {
                uint8_t dim_selector = Data[offset++];
                int64_t tensor_rank = tensor.dim();
                
                if (tensor_rank == 0) {
                    torch::prod(tensor, out_dtype);
                } else {
                    int64_t dim_range = tensor_rank * 2;
                    int64_t dim = (dim_selector % dim_range) - tensor_rank;
                    
                    uint8_t keepdim_selector = 0;
                    if (offset < Size) {
                        keepdim_selector = Data[offset++];
                    }
                    bool keepdim = (keepdim_selector % 2) == 1;
                    
                    torch::prod(tensor, dim, keepdim, out_dtype);
                }
            } else {
                torch::prod(tensor, out_dtype);
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