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
        
        auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }
        
        auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (tensor1.dim() == 0 || tensor2.dim() == 0) {
            return 0;
        }
        
        if (tensor1.dim() == 1) {
            tensor1 = tensor1.unsqueeze(0);
        }
        if (tensor2.dim() == 1) {
            tensor2 = tensor2.unsqueeze(-1);
        }
        
        if (tensor1.dim() > 2) {
            std::vector<int64_t> new_shape = {-1, tensor1.size(-1)};
            tensor1 = tensor1.reshape(new_shape);
        }
        if (tensor2.dim() > 2) {
            std::vector<int64_t> new_shape = {tensor2.size(0), -1};
            tensor2 = tensor2.reshape(new_shape);
        }
        
        auto result = torch::mm(tensor1, tensor2);
        
        if (result.numel() > 0) {
            auto sum = torch::sum(result);
            volatile auto val = sum.item<double>();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}