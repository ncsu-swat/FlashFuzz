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
        
        uint8_t options_byte = Data[offset++];
        
        bool sorted = (options_byte & 0x01) != 0;
        bool return_inverse = (options_byte & 0x02) != 0;
        bool return_counts = (options_byte & 0x04) != 0;
        
        if (options_byte & 0x08) {
            auto result = torch::unique(input_tensor, sorted, return_inverse, return_counts);
        } else {
            if (offset >= Size) {
                auto result = torch::unique(input_tensor, sorted, return_inverse, return_counts);
                return 0;
            }
            
            int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
            int64_t dim = static_cast<int64_t>(dim_raw);
            
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                if (dim < 0) {
                    dim += input_tensor.dim();
                }
            }
            
            auto result = torch::unique_dim(input_tensor, dim, sorted, return_inverse, return_counts);
        }
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                auto result1 = torch::unique_consecutive(input_tensor2, return_inverse, return_counts);
            } catch (...) {
            }
            
            if (input_tensor2.dim() > 0 && offset < Size) {
                int8_t dim_raw2 = static_cast<int8_t>(Data[offset++]);
                int64_t dim2 = static_cast<int64_t>(dim_raw2);
                dim2 = dim2 % input_tensor2.dim();
                if (dim2 < 0) {
                    dim2 += input_tensor2.dim();
                }
                
                try {
                    auto result2 = torch::unique_consecutive(input_tensor2, return_inverse, return_counts, dim2);
                } catch (...) {
                }
            }
        }
        
        if (input_tensor.numel() > 0) {
            auto flattened = input_tensor.flatten();
            auto result_flat = torch::unique(flattened, sorted, return_inverse, return_counts);
        }
        
        if (input_tensor.dim() == 0) {
            auto result_scalar = torch::unique(input_tensor, sorted, return_inverse, return_counts);
        }
        
        if (input_tensor.numel() == 0) {
            auto result_empty = torch::unique(input_tensor, sorted, return_inverse, return_counts);
        }
        
        auto cloned_tensor = input_tensor.clone();
        auto result_clone = torch::unique(cloned_tensor, sorted, return_inverse, return_counts);
        
        if (input_tensor.is_floating_point()) {
            auto nan_tensor = input_tensor.clone();
            if (nan_tensor.numel() > 0) {
                nan_tensor.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                auto result_nan = torch::unique(nan_tensor, sorted, return_inverse, return_counts);
            }
        }
        
        if (input_tensor.is_floating_point()) {
            auto inf_tensor = input_tensor.clone();
            if (inf_tensor.numel() > 0) {
                inf_tensor.flatten()[0] = std::numeric_limits<float>::infinity();
                auto result_inf = torch::unique(inf_tensor, sorted, return_inverse, return_counts);
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