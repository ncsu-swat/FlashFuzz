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
        operation_mode = operation_mode % 4;
        
        if (operation_mode == 0) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            auto result = torch::bitwise_xor(tensor1, tensor2);
        }
        else if (operation_mode == 1) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                int64_t scalar_raw;
                size_t scalar_bytes = sizeof(int64_t);
                if (offset + scalar_bytes <= Size) {
                    std::memcpy(&scalar_raw, Data + offset, scalar_bytes);
                    offset += scalar_bytes;
                } else {
                    scalar_raw = 42;
                }
                
                auto result = torch::bitwise_xor(tensor, scalar_raw);
            }
        }
        else if (operation_mode == 2) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            torch::bitwise_xor_out(tensor1, tensor1, tensor2);
        }
        else if (operation_mode == 3) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            tensor1.bitwise_xor_(tensor2);
        }
        
        if (offset < Size) {
            auto extra_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                auto result1 = torch::bitwise_xor(extra_tensor, extra_tensor);
                
                auto zero_tensor = torch::zeros_like(extra_tensor);
                auto result2 = torch::bitwise_xor(extra_tensor, zero_tensor);
                
                auto ones_tensor = torch::ones_like(extra_tensor);
                if (extra_tensor.dtype() == torch::kBool) {
                    auto result3 = torch::bitwise_xor(extra_tensor, ones_tensor);
                }
                
                if (extra_tensor.numel() > 0) {
                    auto flattened = extra_tensor.flatten();
                    if (flattened.size(0) > 1) {
                        auto first_elem = flattened.slice(0, 0, 1);
                        auto rest_elems = flattened.slice(0, 1, flattened.size(0));
                        if (first_elem.numel() > 0 && rest_elems.numel() > 0) {
                            auto broadcast_result = torch::bitwise_xor(first_elem, rest_elems);
                        }
                    }
                }
                
                auto reshaped = extra_tensor.view({-1});
                if (reshaped.numel() > 0) {
                    auto self_xor = torch::bitwise_xor(reshaped, reshaped);
                }
                
            } catch (...) {
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