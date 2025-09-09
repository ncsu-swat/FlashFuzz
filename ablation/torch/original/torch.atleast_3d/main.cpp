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
        
        uint8_t num_tensors_byte = Data[offset++];
        uint8_t num_tensors = (num_tensors_byte % 5) + 1;
        
        if (num_tensors == 1) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto result = torch::atleast_3d(tensor);
            
            if (result.dim() < 3) {
                throw std::runtime_error("atleast_3d should return tensor with at least 3 dimensions");
            }
        } else {
            std::vector<torch::Tensor> tensors;
            
            for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
                try {
                    auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    tensors.push_back(tensor);
                } catch (const std::exception&) {
                    break;
                }
            }
            
            if (!tensors.empty()) {
                auto results = torch::atleast_3d(tensors);
                
                for (size_t i = 0; i < results.size(); ++i) {
                    if (results[i].dim() < 3) {
                        throw std::runtime_error("atleast_3d should return tensors with at least 3 dimensions");
                    }
                }
            }
        }
        
        if (offset < Size) {
            auto scalar_tensor = torch::tensor(0.5);
            auto result_scalar = torch::atleast_3d(scalar_tensor);
            
            auto empty_tensor = torch::empty({0});
            auto result_empty = torch::atleast_3d(empty_tensor);
            
            auto tensor_1d = torch::arange(5);
            auto result_1d = torch::atleast_3d(tensor_1d);
            
            auto tensor_2d = torch::arange(6).view({2, 3});
            auto result_2d = torch::atleast_3d(tensor_2d);
            
            auto tensor_3d = torch::arange(24).view({2, 3, 4});
            auto result_3d = torch::atleast_3d(tensor_3d);
            
            auto tensor_4d = torch::arange(120).view({2, 3, 4, 5});
            auto result_4d = torch::atleast_3d(tensor_4d);
            
            std::vector<torch::Tensor> mixed_tensors = {scalar_tensor, tensor_1d, tensor_2d};
            auto mixed_results = torch::atleast_3d(mixed_tensors);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}