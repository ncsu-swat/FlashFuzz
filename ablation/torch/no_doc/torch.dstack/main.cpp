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
        
        uint8_t num_tensors_byte = Data[offset++];
        uint8_t num_tensors = (num_tensors_byte % 10) + 1;
        
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);
        
        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) {
                break;
            }
            
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception&) {
                break;
            }
        }
        
        if (tensors.empty()) {
            return 0;
        }
        
        if (tensors.size() == 1) {
            torch::dstack(tensors[0]);
        } else {
            torch::dstack(tensors);
        }
        
        torch::TensorList tensor_list(tensors);
        torch::dstack(tensor_list);
        
        for (const auto& tensor : tensors) {
            if (tensor.dim() == 0) {
                torch::dstack(tensor.unsqueeze(0));
            } else if (tensor.dim() == 1) {
                torch::dstack(tensor.unsqueeze(0).unsqueeze(0));
            } else if (tensor.dim() >= 2) {
                torch::dstack(tensor);
            }
        }
        
        std::vector<torch::Tensor> mixed_tensors;
        for (size_t i = 0; i < tensors.size() && i < 3; ++i) {
            auto t = tensors[i];
            if (t.dim() == 0) {
                mixed_tensors.push_back(t.unsqueeze(0).unsqueeze(0));
            } else if (t.dim() == 1) {
                mixed_tensors.push_back(t.unsqueeze(0));
            } else {
                mixed_tensors.push_back(t);
            }
        }
        
        if (!mixed_tensors.empty()) {
            torch::dstack(mixed_tensors);
        }
        
        if (tensors.size() >= 2) {
            std::vector<torch::Tensor> same_shape_tensors;
            auto base_tensor = tensors[0];
            if (base_tensor.dim() >= 2) {
                same_shape_tensors.push_back(base_tensor);
                for (size_t i = 1; i < tensors.size() && same_shape_tensors.size() < 5; ++i) {
                    try {
                        auto reshaped = tensors[i].reshape(base_tensor.sizes());
                        same_shape_tensors.push_back(reshaped);
                    } catch (const std::exception&) {
                        continue;
                    }
                }
                
                if (same_shape_tensors.size() >= 2) {
                    torch::dstack(same_shape_tensors);
                }
            }
        }
        
        for (const auto& tensor : tensors) {
            if (tensor.numel() > 0) {
                auto squeezed = tensor.squeeze();
                torch::dstack(squeezed);
                
                if (tensor.dim() > 0) {
                    auto flattened = tensor.flatten();
                    torch::dstack(flattened);
                }
            }
        }
        
        if (!tensors.empty()) {
            auto first_tensor = tensors[0];
            std::vector<torch::Tensor> broadcast_tensors;
            broadcast_tensors.push_back(first_tensor);
            
            for (size_t i = 1; i < tensors.size() && broadcast_tensors.size() < 3; ++i) {
                try {
                    auto broadcasted = torch::broadcast_to(tensors[i], first_tensor.sizes());
                    broadcast_tensors.push_back(broadcasted);
                } catch (const std::exception&) {
                    continue;
                }
            }
            
            if (broadcast_tensors.size() >= 2) {
                torch::dstack(broadcast_tensors);
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