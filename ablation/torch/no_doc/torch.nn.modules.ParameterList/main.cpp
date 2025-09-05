#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine number of parameters
        if (Size < 1) {
            return 0;
        }
        
        // Create a ParameterList
        torch::nn::ParameterList param_list;
        
        // Determine number of parameters to create (0-10)
        uint8_t num_params = Data[offset++] % 11;
        
        // Create and add parameters with diverse shapes and properties
        for (uint8_t i = 0; i < num_params && offset < Size; ++i) {
            try {
                // Create a tensor from fuzzer input
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Set requires_grad based on fuzzer input if available
                if (offset < Size) {
                    bool requires_grad = Data[offset++] & 1;
                    tensor.set_requires_grad(requires_grad);
                }
                
                // Create parameter from tensor
                auto param = torch::nn::Parameter(tensor);
                
                // Append to ParameterList
                param_list->append(param);
                
            } catch (const std::exception& e) {
                // Continue with next parameter if this one fails
                continue;
            }
        }
        
        // Exercise various ParameterList operations
        
        // 1. Size operations
        auto list_size = param_list->size();
        bool is_empty = param_list->is_empty();
        
        // 2. Access operations (if not empty)
        if (!is_empty && offset < Size) {
            // Random access by index
            size_t access_idx = (offset < Size ? Data[offset++] : 0) % list_size;
            try {
                auto accessed_param = param_list[access_idx];
                
                // Perform operations on accessed parameter
                if (accessed_param.defined()) {
                    auto grad_status = accessed_param.requires_grad();
                    auto dtype = accessed_param.dtype();
                    auto shape = accessed_param.sizes();
                    
                    // Try gradient computation if requires_grad
                    if (grad_status && accessed_param.numel() > 0) {
                        try {
                            auto sum_val = accessed_param.sum();
                            if (sum_val.requires_grad()) {
                                sum_val.backward();
                            }
                        } catch (...) {
                            // Ignore gradient computation failures
                        }
                    }
                }
            } catch (...) {
                // Ignore access failures
            }
        }
        
        // 3. Iteration through parameters
        try {
            for (const auto& param : *param_list) {
                if (param.defined()) {
                    auto numel = param.numel();
                    auto device = param.device();
                    
                    // Try some tensor operations
                    if (numel > 0 && offset < Size) {
                        uint8_t op_selector = (offset < Size ? Data[offset++] : 0) % 5;
                        switch (op_selector) {
                            case 0:
                                // Clone
                                auto cloned = param.clone();
                                break;
                            case 1:
                                // Transpose if 2D or higher
                                if (param.dim() >= 2) {
                                    auto transposed = param.transpose(0, 1);
                                }
                                break;
                            case 2:
                                // Reshape
                                auto reshaped = param.reshape({-1});
                                break;
                            case 3:
                                // Type conversion
                                if (param.dtype() != torch::kFloat32) {
                                    auto converted = param.to(torch::kFloat32);
                                }
                                break;
                            case 4:
                                // Contiguous
                                auto contig = param.contiguous();
                                break;
                        }
                    }
                }
            }
        } catch (...) {
            // Ignore iteration failures
        }
        
        // 4. Extend with another ParameterList if we have more data
        if (offset < Size) {
            try {
                torch::nn::ParameterList other_list;
                
                uint8_t num_other = (Data[offset++] % 5);
                for (uint8_t i = 0; i < num_other && offset < Size; ++i) {
                    try {
                        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        other_list->append(torch::nn::Parameter(tensor));
                    } catch (...) {
                        continue;
                    }
                }
                
                // Extend original list
                param_list->extend(*other_list);
                
            } catch (...) {
                // Ignore extend failures
            }
        }
        
        // 5. Named parameters access
        try {
            auto named_params = param_list->named_parameters();
            for (const auto& named_param : named_params) {
                auto name = named_param.key();
                auto param = named_param.value();
                
                if (param.defined()) {
                    auto is_leaf = param.is_leaf();
                    auto is_cuda = param.is_cuda();
                }
            }
        } catch (...) {
            // Ignore named parameters access failures
        }
        
        // 6. Parameters vector access
        try {
            auto params_vec = param_list->parameters();
            
            // Modify parameters if we have data
            if (!params_vec.empty() && offset < Size) {
                size_t modify_idx = (Data[offset++] % params_vec.size());
                
                // Try in-place operations
                if (params_vec[modify_idx].defined() && params_vec[modify_idx].numel() > 0) {
                    try {
                        params_vec[modify_idx].zero_();
                    } catch (...) {}
                    
                    try {
                        params_vec[modify_idx].fill_(1.0);
                    } catch (...) {}
                    
                    try {
                        params_vec[modify_idx].uniform_(-1.0, 1.0);
                    } catch (...) {}
                }
            }
        } catch (...) {
            // Ignore parameters vector failures
        }
        
        // 7. Clone the entire module
        try {
            auto cloned_list = param_list->clone();
            
            // Verify clone worked
            if (cloned_list->size() != param_list->size()) {
                std::cerr << "Clone size mismatch" << std::endl;
            }
        } catch (...) {
            // Ignore clone failures
        }
        
        // 8. Reset if we have specific fuzzer input
        if (offset < Size && (Data[offset++] & 1)) {
            try {
                param_list->reset();
            } catch (...) {
                // Ignore reset failures
            }
        }
        
        // 9. Pretty print (exercises string conversion)
        try {
            param_list->pretty_print(std::cout);
            std::cout << std::endl;
        } catch (...) {
            // Ignore print failures
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}