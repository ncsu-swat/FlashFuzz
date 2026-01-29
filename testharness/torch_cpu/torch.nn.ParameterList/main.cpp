#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Determine number of parameters to create
        if (Size < 1) return 0;
        uint8_t num_params = Data[offset++] % 10 + 1; // 1-10 parameters
        
        // Create a ParameterList
        torch::nn::ParameterList param_list;
        
        // Add parameters to the list
        for (uint8_t i = 0; i < num_params && offset < Size; ++i) {
            // Create a tensor to use as parameter
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (!tensor.defined()) continue;
            
            // Ensure tensor requires grad for it to be a proper parameter
            tensor = tensor.to(torch::kFloat).requires_grad_(true);
            
            // Add tensor as parameter to list
            param_list->append(tensor);
        }
        
        // Test various operations on the ParameterList
        if (param_list->size() > 0) {
            // Access parameters by index using at()
            for (size_t i = 0; i < param_list->size(); ++i) {
                torch::Tensor param = param_list[i];
                
                // Try some operations on the parameter
                if (param.defined() && param.numel() > 0) {
                    torch::Tensor result = param.clone();
                    if (param.is_floating_point()) {
                        result = param * 2.0;
                    }
                }
            }
            
            // Test parameters() method
            auto params = param_list->parameters();
            for (const auto& param : params) {
                if (param.defined()) {
                    torch::Tensor result = param.clone();
                }
            }
            
            // Test append
            if (offset < Size) {
                torch::Tensor new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (new_tensor.defined()) {
                    new_tensor = new_tensor.to(torch::kFloat).requires_grad_(true);
                    param_list->append(new_tensor);
                }
            }
            
            // Test extend with another ParameterList
            if (offset < Size) {
                torch::nn::ParameterList second_list;
                uint8_t second_size = (offset < Size) ? (Data[offset++] % 5 + 1) : 1;
                
                for (uint8_t i = 0; i < second_size && offset < Size; ++i) {
                    torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    if (tensor.defined()) {
                        tensor = tensor.to(torch::kFloat).requires_grad_(true);
                        second_list->append(tensor);
                    }
                }
                
                if (second_list->size() > 0) {
                    param_list->extend(*second_list);
                }
            }
            
            // Test named_parameters
            auto named_params = param_list->named_parameters();
            for (const auto& pair : named_params) {
                const torch::Tensor& param = pair.value();
                
                if (param.defined() && param.numel() > 0) {
                    torch::Tensor result = param.clone();
                }
            }
            
            // Test is_empty
            bool is_empty = param_list->is_empty();
            (void)is_empty;
        }
        
        // Test empty ParameterList
        torch::nn::ParameterList empty_list;
        bool empty_is_empty = empty_list->is_empty();
        (void)empty_is_empty;
        auto empty_params = empty_list->parameters();
        
        // Test with out-of-bounds access (should throw)
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                if (param_list->size() > 0) {
                    size_t invalid_idx = param_list->size() + 10;
                    torch::Tensor invalid_param = param_list[invalid_idx];
                    (void)invalid_param;
                }
            } catch (const std::exception&) {
                // Expected exception for out-of-bounds access
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}