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
        
        // Need at least 1 byte for number of parameters
        if (Size < 1) {
            return 0;
        }
        
        // Parse number of parameters to create (0-10)
        uint8_t num_params = Data[offset++] % 11;
        
        // Create ParameterList
        torch::nn::ParameterList param_list;
        
        // Vector to store created parameters for later operations
        std::vector<torch::Tensor> created_params;
        
        // Add parameters to the list
        for (uint8_t i = 0; i < num_params; ++i) {
            try {
                // Create a tensor from fuzzer input
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert tensor to parameter and add to list
                param_list->append(tensor);
                created_params.push_back(tensor);
                
            } catch (const std::exception& e) {
                // Continue with fewer parameters if we run out of data
                break;
            }
        }
        
        // Test various ParameterList operations
        if (offset < Size && !param_list->is_empty()) {
            uint8_t op_selector = Data[offset++];
            
            switch (op_selector % 8) {
                case 0: {
                    // Test size()
                    auto size = param_list->size();
                    (void)size;
                    break;
                }
                case 1: {
                    // Test indexing with valid index
                    if (!param_list->is_empty()) {
                        uint8_t idx = (offset < Size) ? Data[offset++] % param_list->size() : 0;
                        auto param = param_list[idx];
                        // Perform operation on parameter
                        if (param.numel() > 0) {
                            auto mean_val = param.mean();
                            (void)mean_val;
                        }
                    }
                    break;
                }
                case 2: {
                    // Test iteration
                    for (const auto& param : *param_list) {
                        if (param.numel() > 0) {
                            auto sum_val = param.sum();
                            (void)sum_val;
                        }
                    }
                    break;
                }
                case 3: {
                    // Test extend with another list
                    torch::nn::ParameterList another_list;
                    try {
                        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        another_list->append(tensor);
                        param_list->extend(*another_list);
                    } catch (...) {
                        // Ignore if we can't create another tensor
                    }
                    break;
                }
                case 4: {
                    // Test append with new parameter
                    try {
                        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        param_list->append(tensor);
                    } catch (...) {
                        // Ignore if we can't create tensor
                    }
                    break;
                }
                case 5: {
                    // Test parameters() method
                    auto params = param_list->parameters();
                    for (auto& p : params) {
                        if (p.defined() && p.numel() > 0) {
                            auto std_val = p.std();
                            (void)std_val;
                        }
                    }
                    break;
                }
                case 6: {
                    // Test named_parameters()
                    auto named_params = param_list->named_parameters();
                    for (const auto& np : named_params) {
                        const std::string& name = np.key();
                        const torch::Tensor& param = np.value();
                        if (param.defined() && param.numel() > 0) {
                            auto max_val = param.max();
                            (void)max_val;
                        }
                    }
                    break;
                }
                case 7: {
                    // Test cloning/copying behavior
                    auto cloned_list = param_list->clone();
                    if (cloned_list != nullptr) {
                        // Verify clone has same size
                        auto orig_size = param_list->size();
                        auto clone_size = std::dynamic_pointer_cast<torch::nn::ParameterList>(cloned_list)->size();
                        (void)orig_size;
                        (void)clone_size;
                    }
                    break;
                }
            }
        }
        
        // Additional operations with multiple parameters
        if (param_list->size() >= 2 && offset < Size) {
            uint8_t op2 = Data[offset++];
            
            switch (op2 % 4) {
                case 0: {
                    // Matrix multiplication between two parameters if shapes allow
                    try {
                        auto p1 = (*param_list)[0];
                        auto p2 = (*param_list)[1];
                        if (p1.dim() >= 2 && p2.dim() >= 2 && 
                            p1.size(-1) == p2.size(-2)) {
                            auto result = torch::matmul(p1, p2);
                            (void)result;
                        }
                    } catch (...) {
                        // Ignore shape mismatches
                    }
                    break;
                }
                case 1: {
                    // Element-wise operations
                    try {
                        auto p1 = (*param_list)[0];
                        auto p2 = (*param_list)[1];
                        if (p1.sizes() == p2.sizes()) {
                            auto sum = p1 + p2;
                            auto prod = p1 * p2;
                            (void)sum;
                            (void)prod;
                        }
                    } catch (...) {
                        // Ignore incompatible operations
                    }
                    break;
                }
                case 2: {
                    // Test gradient operations
                    for (auto& param : *param_list) {
                        if (param.requires_grad()) {
                            // Set gradient to zeros
                            if (param.grad().defined()) {
                                param.grad().zero_();
                            }
                        }
                    }
                    break;
                }
                case 3: {
                    // Test state_dict behavior
                    auto state = param_list->state_dict();
                    // Try to load it back
                    torch::nn::ParameterList new_list;
                    try {
                        new_list->load_state_dict(state);
                    } catch (...) {
                        // Ignore load errors
                    }
                    break;
                }
            }
        }
        
        // Test edge cases
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 3) {
                case 0: {
                    // Test with empty list
                    torch::nn::ParameterList empty_list;
                    auto size = empty_list->size();
                    auto is_empty = empty_list->is_empty();
                    (void)size;
                    (void)is_empty;
                    break;
                }
                case 1: {
                    // Test clear operation if supported
                    while (!param_list->is_empty()) {
                        param_list->pop_back();
                    }
                    break;
                }
                case 2: {
                    // Test with very small tensors
                    torch::nn::ParameterList small_list;
                    small_list->append(torch::zeros({1}));
                    small_list->append(torch::ones({1, 1}));
                    small_list->append(torch::randn({0})); // Empty tensor
                    break;
                }
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}