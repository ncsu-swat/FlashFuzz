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
            torch::count_nonzero(input_tensor);
            return 0;
        }
        
        uint8_t dim_mode = Data[offset++];
        
        if (dim_mode % 3 == 0) {
            torch::count_nonzero(input_tensor);
        } else if (dim_mode % 3 == 1) {
            if (offset >= Size) {
                return 0;
            }
            
            int64_t dim_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                dim_raw = static_cast<int64_t>(Data[offset++]);
            }
            
            int64_t tensor_ndim = input_tensor.dim();
            int64_t dim = dim_raw;
            
            if (tensor_ndim > 0) {
                dim = dim % (2 * tensor_ndim) - tensor_ndim;
            }
            
            torch::count_nonzero(input_tensor, dim);
        } else {
            std::vector<int64_t> dims;
            uint8_t num_dims = 1;
            
            if (offset < Size) {
                num_dims = (Data[offset++] % 5) + 1;
            }
            
            int64_t tensor_ndim = input_tensor.dim();
            
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int64_t dim_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else if (offset < Size) {
                    dim_raw = static_cast<int64_t>(Data[offset++]);
                } else {
                    break;
                }
                
                int64_t dim = dim_raw;
                if (tensor_ndim > 0) {
                    dim = dim % (2 * tensor_ndim) - tensor_ndim;
                }
                dims.push_back(dim);
            }
            
            if (!dims.empty()) {
                torch::count_nonzero(input_tensor, dims);
            }
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        torch::count_nonzero(zero_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        torch::count_nonzero(ones_tensor);
        
        if (input_tensor.numel() > 0) {
            auto mixed_tensor = input_tensor.clone();
            if (mixed_tensor.is_floating_point()) {
                mixed_tensor = mixed_tensor * 0.0;
                mixed_tensor.index_put_({0}, 1.0);
            } else if (mixed_tensor.dtype() == torch::kBool) {
                mixed_tensor.fill_(false);
                mixed_tensor.index_put_({0}, true);
            } else {
                mixed_tensor.fill_(0);
                mixed_tensor.index_put_({0}, 1);
            }
            torch::count_nonzero(mixed_tensor);
        }
        
        if (input_tensor.dim() > 0) {
            for (int64_t d = -input_tensor.dim(); d < input_tensor.dim(); ++d) {
                torch::count_nonzero(input_tensor, d);
            }
        }
        
        auto empty_tensor = torch::empty({0}, input_tensor.options());
        torch::count_nonzero(empty_tensor);
        
        if (input_tensor.dtype() != torch::kBool) {
            auto bool_tensor = input_tensor.to(torch::kBool);
            torch::count_nonzero(bool_tensor);
        }
        
        if (input_tensor.is_floating_point()) {
            auto inf_tensor = input_tensor.clone();
            if (inf_tensor.numel() > 0) {
                inf_tensor.index_put_({0}, std::numeric_limits<double>::infinity());
                torch::count_nonzero(inf_tensor);
                
                inf_tensor.index_put_({0}, -std::numeric_limits<double>::infinity());
                torch::count_nonzero(inf_tensor);
                
                inf_tensor.index_put_({0}, std::numeric_limits<double>::quiet_NaN());
                torch::count_nonzero(inf_tensor);
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