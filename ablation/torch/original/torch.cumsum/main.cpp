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
            return 0;
        }
        
        uint8_t dim_byte = Data[offset++];
        int64_t dim = static_cast<int64_t>(static_cast<int8_t>(dim_byte));
        
        torch::cumsum(input_tensor, dim);
        
        if (input_tensor.dim() > 0) {
            int64_t valid_dim = dim % input_tensor.dim();
            torch::cumsum(input_tensor, valid_dim);
            
            torch::cumsum(input_tensor, -1);
            torch::cumsum(input_tensor, input_tensor.dim() - 1);
        }
        
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto target_dtype = fuzzer_utils::parseDataType(dtype_selector);
            torch::cumsum(input_tensor, 0, target_dtype);
        }
        
        torch::cumsum(input_tensor, 0);
        
        if (input_tensor.dim() > 1) {
            torch::cumsum(input_tensor, 1);
        }
        
        torch::cumsum(input_tensor, -input_tensor.dim());
        
        int64_t large_dim = 1000000;
        torch::cumsum(input_tensor, large_dim);
        
        int64_t negative_large_dim = -1000000;
        torch::cumsum(input_tensor, negative_large_dim);
        
        if (input_tensor.numel() > 0) {
            auto out_tensor = torch::empty_like(input_tensor);
            torch::cumsum_out(out_tensor, input_tensor, 0);
        }
        
        auto float_tensor = input_tensor.to(torch::kFloat);
        torch::cumsum(float_tensor, 0);
        
        auto double_tensor = input_tensor.to(torch::kDouble);
        torch::cumsum(double_tensor, 0);
        
        if (input_tensor.dtype() != torch::kBool) {
            auto int_tensor = input_tensor.to(torch::kInt64);
            torch::cumsum(int_tensor, 0);
        }
        
        for (int64_t d = -input_tensor.dim(); d < input_tensor.dim(); ++d) {
            torch::cumsum(input_tensor, d);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}