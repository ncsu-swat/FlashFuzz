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
        
        uint8_t operation_mode = Data[offset++];
        
        if (operation_mode % 4 == 0) {
            torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            torch::Tensor result = torch::ne(tensor1, tensor2);
            
            if (offset < Size) {
                torch::Tensor broadcasted_result = tensor1.ne(tensor2);
            }
        }
        else if (operation_mode % 4 == 1) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset + sizeof(double) <= Size) {
                double scalar_val;
                std::memcpy(&scalar_val, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                torch::Tensor result = torch::ne(tensor, scalar_val);
                torch::Tensor result2 = tensor.ne(scalar_val);
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                int64_t int_val;
                std::memcpy(&int_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                torch::Tensor result = torch::ne(tensor, int_val);
            }
        }
        else if (operation_mode % 4 == 2) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                torch::Scalar scalar_from_byte = static_cast<float>(Data[offset++]);
                torch::Tensor result = torch::ne(tensor, scalar_from_byte);
            }
            
            if (tensor.numel() > 0) {
                torch::Tensor self_compare = torch::ne(tensor, tensor);
            }
        }
        else {
            torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                auto tensor1_clone = tensor1.clone();
                torch::Tensor result = torch::ne(tensor1, tensor1_clone);
                
                if (tensor1.numel() > 0) {
                    auto modified_tensor = tensor1 + 1e-10;
                    torch::Tensor result2 = torch::ne(tensor1, modified_tensor);
                }
            }
            
            if (offset < Size) {
                auto reshaped = tensor1.view(-1);
                if (reshaped.numel() == tensor1.numel()) {
                    torch::Tensor result = torch::ne(tensor1, reshaped);
                }
            }
        }
        
        if (offset + 1 < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor another_empty = torch::empty({0});
            torch::Tensor empty_result = torch::ne(empty_tensor, another_empty);
        }
        
        if (offset + 2 < Size) {
            torch::Tensor zero_dim = torch::tensor(42.0);
            torch::Tensor another_zero_dim = torch::tensor(42.0);
            torch::Tensor zero_dim_result = torch::ne(zero_dim, another_zero_dim);
            
            torch::Tensor different_zero_dim = torch::tensor(43.0);
            torch::Tensor diff_result = torch::ne(zero_dim, different_zero_dim);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}