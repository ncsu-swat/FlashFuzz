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
        
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::Tensor result = torch::logical_and(tensor1, tensor2);
        
        torch::Tensor scalar_tensor = torch::tensor(true);
        torch::Tensor result_with_scalar = torch::logical_and(tensor1, scalar_tensor);
        
        torch::Tensor bool_tensor1 = tensor1.to(torch::kBool);
        torch::Tensor bool_tensor2 = tensor2.to(torch::kBool);
        torch::Tensor bool_result = torch::logical_and(bool_tensor1, bool_tensor2);
        
        torch::Tensor empty_tensor = torch::empty({0}, torch::kBool);
        if (tensor1.numel() == 0) {
            torch::Tensor empty_result = torch::logical_and(empty_tensor, empty_tensor);
        }
        
        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
            try {
                torch::Tensor broadcast_result = torch::logical_and(tensor1, tensor2);
            } catch (...) {
            }
        }
        
        torch::Tensor zero_tensor = torch::zeros_like(tensor1);
        torch::Tensor ones_tensor = torch::ones_like(tensor1);
        torch::Tensor mixed_result1 = torch::logical_and(zero_tensor, ones_tensor);
        torch::Tensor mixed_result2 = torch::logical_and(ones_tensor, zero_tensor);
        torch::Tensor mixed_result3 = torch::logical_and(ones_tensor, ones_tensor);
        
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 0) {
                torch::Tensor inplace_tensor = tensor1.clone();
                inplace_tensor.logical_and_(tensor2);
            }
        }
        
        torch::Tensor large_tensor = torch::ones({1000, 1000}, torch::kBool);
        torch::Tensor small_tensor = torch::zeros({1}, torch::kBool);
        try {
            torch::Tensor broadcast_large = torch::logical_and(large_tensor, small_tensor);
        } catch (...) {
        }
        
        std::vector<torch::Tensor> tensor_list = {tensor1, tensor2, bool_tensor1, bool_tensor2};
        for (size_t i = 0; i < tensor_list.size(); ++i) {
            for (size_t j = i + 1; j < tensor_list.size(); ++j) {
                try {
                    torch::Tensor cross_result = torch::logical_and(tensor_list[i], tensor_list[j]);
                } catch (...) {
                }
            }
        }
        
        if (tensor1.is_floating_point()) {
            torch::Tensor inf_tensor = torch::full_like(tensor1, std::numeric_limits<float>::infinity());
            torch::Tensor nan_tensor = torch::full_like(tensor1, std::numeric_limits<float>::quiet_NaN());
            try {
                torch::Tensor inf_result = torch::logical_and(inf_tensor, tensor2);
                torch::Tensor nan_result = torch::logical_and(nan_tensor, tensor2);
            } catch (...) {
            }
        }
        
        if (tensor1.is_complex()) {
            torch::Tensor complex_zero = torch::zeros_like(tensor1);
            torch::Tensor complex_result = torch::logical_and(tensor1, complex_zero);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}