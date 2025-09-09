#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        int64_t result = tensor.numel();
        
        if (offset < Size) {
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t result2 = tensor2.numel();
        }
        
        torch::Tensor empty_tensor = torch::empty({});
        int64_t empty_result = empty_tensor.numel();
        
        torch::Tensor zero_dim_tensor = torch::empty({0});
        int64_t zero_dim_result = zero_dim_tensor.numel();
        
        torch::Tensor multi_zero_tensor = torch::empty({0, 5, 0});
        int64_t multi_zero_result = multi_zero_tensor.numel();
        
        torch::Tensor large_tensor = torch::empty({1000, 1000});
        int64_t large_result = large_tensor.numel();
        
        if (offset + 8 <= Size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, 8);
            offset += 8;
            
            if (raw_dim > 0 && raw_dim < 10000) {
                torch::Tensor dynamic_tensor = torch::empty({raw_dim});
                int64_t dynamic_result = dynamic_tensor.numel();
            }
        }
        
        if (offset + 16 <= Size) {
            int64_t dim1, dim2;
            std::memcpy(&dim1, Data + offset, 8);
            std::memcpy(&dim2, Data + offset + 8, 8);
            offset += 16;
            
            dim1 = std::abs(dim1) % 1000 + 1;
            dim2 = std::abs(dim2) % 1000 + 1;
            
            torch::Tensor two_d_tensor = torch::empty({dim1, dim2});
            int64_t two_d_result = two_d_tensor.numel();
        }
        
        std::vector<torch::ScalarType> types = {
            torch::kFloat, torch::kDouble, torch::kInt32, torch::kInt64,
            torch::kBool, torch::kUInt8, torch::kInt8, torch::kComplexFloat
        };
        
        for (auto dtype : types) {
            torch::Tensor typed_tensor = torch::empty({10, 10}, torch::TensorOptions().dtype(dtype));
            int64_t typed_result = typed_tensor.numel();
        }
        
        torch::Tensor view_tensor = torch::empty({2, 3, 4});
        torch::Tensor reshaped = view_tensor.view({6, 4});
        int64_t view_result = reshaped.numel();
        
        torch::Tensor slice_tensor = torch::empty({100, 100});
        torch::Tensor sliced = slice_tensor.slice(0, 10, 50);
        int64_t slice_result = sliced.numel();
        
        if (tensor.dim() > 0) {
            torch::Tensor squeezed = tensor.squeeze();
            int64_t squeeze_result = squeezed.numel();
        }
        
        torch::Tensor unsqueezed = tensor.unsqueeze(0);
        int64_t unsqueeze_result = unsqueezed.numel();
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}