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
        int64_t dim = static_cast<int64_t>(dim_byte) % (input_tensor.dim() + 1);
        if (input_tensor.dim() > 0 && dim_byte % 2 == 1) {
            dim = -dim - 1;
        }
        
        torch::cumsum(input_tensor, dim);
        
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            auto target_dtype = fuzzer_utils::parseDataType(dtype_byte);
            torch::cumsum(input_tensor, dim, target_dtype);
        }
        
        if (input_tensor.dim() > 0) {
            for (int64_t d = 0; d < input_tensor.dim(); ++d) {
                torch::cumsum(input_tensor, d);
            }
            for (int64_t d = -input_tensor.dim(); d < 0; ++d) {
                torch::cumsum(input_tensor, d);
            }
        }
        
        if (input_tensor.numel() > 0) {
            torch::cumsum(input_tensor.flatten(), 0);
        }
        
        auto large_dim = static_cast<int64_t>(input_tensor.dim()) + 100;
        torch::cumsum(input_tensor, large_dim);
        
        auto negative_large_dim = -large_dim;
        torch::cumsum(input_tensor, negative_large_dim);
        
        if (input_tensor.dim() == 0) {
            torch::cumsum(input_tensor, 0);
            torch::cumsum(input_tensor, -1);
            torch::cumsum(input_tensor, 1);
        }
        
        auto reshaped = input_tensor.view({-1});
        torch::cumsum(reshaped, 0);
        
        if (input_tensor.numel() > 1) {
            auto squeezed = input_tensor.squeeze();
            if (squeezed.dim() > 0) {
                torch::cumsum(squeezed, 0);
            }
        }
        
        std::vector<torch::ScalarType> test_dtypes = {
            torch::kFloat, torch::kDouble, torch::kInt32, torch::kInt64, torch::kBool
        };
        
        for (auto dtype : test_dtypes) {
            if (dtype != input_tensor.scalar_type()) {
                torch::cumsum(input_tensor, 0, dtype);
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