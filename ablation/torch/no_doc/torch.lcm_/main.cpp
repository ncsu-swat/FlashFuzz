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
        auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (tensor1.numel() == 0 || tensor2.numel() == 0) {
            return 0;
        }
        
        auto dtype1 = tensor1.dtype();
        auto dtype2 = tensor2.dtype();
        
        if (dtype1 == torch::kComplexFloat || dtype1 == torch::kComplexDouble ||
            dtype2 == torch::kComplexFloat || dtype2 == torch::kComplexDouble ||
            dtype1 == torch::kHalf || dtype1 == torch::kBFloat16 ||
            dtype2 == torch::kHalf || dtype2 == torch::kBFloat16 ||
            dtype1 == torch::kBool || dtype2 == torch::kBool) {
            return 0;
        }
        
        if (dtype1 == torch::kFloat || dtype1 == torch::kDouble) {
            tensor1 = tensor1.to(torch::kInt64);
        }
        if (dtype2 == torch::kFloat || dtype2 == torch::kDouble) {
            tensor2 = tensor2.to(torch::kInt64);
        }
        
        try {
            tensor2 = tensor2.broadcast_to(tensor1.sizes());
        } catch (...) {
            try {
                tensor1 = tensor1.broadcast_to(tensor2.sizes());
            } catch (...) {
                return 0;
            }
        }
        
        auto original_tensor1 = tensor1.clone();
        
        tensor1.lcm_(tensor2);
        
        if (offset < Size) {
            auto tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            if (tensor3.numel() > 0) {
                auto dtype3 = tensor3.dtype();
                if (dtype3 != torch::kComplexFloat && dtype3 != torch::kComplexDouble &&
                    dtype3 != torch::kHalf && dtype3 != torch::kBFloat16 && dtype3 != torch::kBool) {
                    if (dtype3 == torch::kFloat || dtype3 == torch::kDouble) {
                        tensor3 = tensor3.to(torch::kInt64);
                    }
                    try {
                        tensor3 = tensor3.broadcast_to(tensor1.sizes());
                        tensor1.lcm_(tensor3);
                    } catch (...) {
                    }
                }
            }
        }
        
        auto zero_tensor = torch::zeros_like(original_tensor1);
        try {
            zero_tensor.lcm_(original_tensor1);
        } catch (...) {
        }
        
        auto ones_tensor = torch::ones_like(original_tensor1);
        try {
            ones_tensor.lcm_(original_tensor1);
        } catch (...) {
        }
        
        auto negative_tensor = -torch::abs(original_tensor1);
        try {
            negative_tensor.lcm_(original_tensor1);
        } catch (...) {
        }
        
        auto large_tensor = original_tensor1 + 1000000;
        try {
            large_tensor.lcm_(original_tensor1);
        } catch (...) {
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}