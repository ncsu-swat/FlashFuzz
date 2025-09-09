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
            torch::nanmean(input_tensor);
            return 0;
        }
        
        uint8_t operation_selector = Data[offset++];
        uint8_t operation_type = operation_selector % 4;
        
        if (operation_type == 0) {
            torch::nanmean(input_tensor);
        }
        else if (operation_type == 1) {
            if (offset >= Size) {
                return 0;
            }
            uint8_t dim_selector = Data[offset++];
            int64_t tensor_rank = input_tensor.dim();
            if (tensor_rank > 0) {
                int64_t dim = static_cast<int64_t>(dim_selector) % (tensor_rank * 2) - tensor_rank;
                torch::nanmean(input_tensor, dim);
            } else {
                torch::nanmean(input_tensor);
            }
        }
        else if (operation_type == 2) {
            if (offset >= Size) {
                return 0;
            }
            uint8_t dim_selector = Data[offset++];
            uint8_t keepdim_selector = Data[offset < Size ? offset++ : offset];
            int64_t tensor_rank = input_tensor.dim();
            bool keepdim = (keepdim_selector % 2) == 1;
            if (tensor_rank > 0) {
                int64_t dim = static_cast<int64_t>(dim_selector) % (tensor_rank * 2) - tensor_rank;
                torch::nanmean(input_tensor, dim, keepdim);
            } else {
                torch::nanmean(input_tensor, c10::nullopt, keepdim);
            }
        }
        else {
            if (offset + 1 >= Size) {
                return 0;
            }
            uint8_t num_dims_selector = Data[offset++];
            uint8_t keepdim_selector = Data[offset++];
            int64_t tensor_rank = input_tensor.dim();
            bool keepdim = (keepdim_selector % 2) == 1;
            
            if (tensor_rank == 0) {
                torch::nanmean(input_tensor, c10::nullopt, keepdim);
            } else {
                uint8_t num_dims = (num_dims_selector % tensor_rank) + 1;
                std::vector<int64_t> dims;
                
                for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                    uint8_t dim_byte = Data[offset++];
                    int64_t dim = static_cast<int64_t>(dim_byte) % (tensor_rank * 2) - tensor_rank;
                    dims.push_back(dim);
                }
                
                if (dims.empty()) {
                    torch::nanmean(input_tensor);
                } else {
                    torch::nanmean(input_tensor, dims, keepdim);
                }
            }
        }
        
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType out_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            if (operation_type == 0) {
                torch::nanmean(input_tensor, out_dtype);
            }
            else if (operation_type == 1) {
                if (offset >= Size) {
                    return 0;
                }
                uint8_t dim_selector = Data[offset++];
                int64_t tensor_rank = input_tensor.dim();
                if (tensor_rank > 0) {
                    int64_t dim = static_cast<int64_t>(dim_selector) % (tensor_rank * 2) - tensor_rank;
                    torch::nanmean(input_tensor, dim, false, out_dtype);
                } else {
                    torch::nanmean(input_tensor, out_dtype);
                }
            }
        }
        
        auto empty_tensor = torch::empty({0});
        torch::nanmean(empty_tensor);
        
        auto nan_tensor = torch::full({3, 3}, std::numeric_limits<double>::quiet_NaN());
        torch::nanmean(nan_tensor);
        
        auto mixed_tensor = torch::tensor({1.0, std::numeric_limits<double>::quiet_NaN(), 3.0});
        torch::nanmean(mixed_tensor);
        
        if (input_tensor.numel() > 0) {
            auto inf_tensor = input_tensor.clone();
            if (inf_tensor.dtype() == torch::kFloat || inf_tensor.dtype() == torch::kDouble) {
                inf_tensor.fill_(std::numeric_limits<double>::infinity());
                torch::nanmean(inf_tensor);
                
                inf_tensor.fill_(-std::numeric_limits<double>::infinity());
                torch::nanmean(inf_tensor);
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