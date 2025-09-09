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
        
        int64_t result = torch::numel(tensor);
        
        if (offset < Size) {
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t result2 = torch::numel(tensor2);
        }
        
        if (offset < Size) {
            auto tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t result3 = torch::numel(tensor3);
        }
        
        torch::Tensor empty_tensor = torch::empty({});
        int64_t empty_result = torch::numel(empty_tensor);
        
        torch::Tensor zero_dim_tensor = torch::empty({0});
        int64_t zero_dim_result = torch::numel(zero_dim_tensor);
        
        torch::Tensor multi_zero_tensor = torch::empty({0, 5, 0});
        int64_t multi_zero_result = torch::numel(multi_zero_tensor);
        
        torch::Tensor large_tensor = torch::empty({1000, 1000});
        int64_t large_result = torch::numel(large_tensor);
        
        torch::Tensor complex_shape = torch::empty({1, 2, 3, 4, 5});
        int64_t complex_result = torch::numel(complex_shape);
        
        torch::Tensor single_element = torch::empty({1});
        int64_t single_result = torch::numel(single_element);
        
        torch::Tensor high_dim = torch::empty({2, 2, 2, 2});
        int64_t high_dim_result = torch::numel(high_dim);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}