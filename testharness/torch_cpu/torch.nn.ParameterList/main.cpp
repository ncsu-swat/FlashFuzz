#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
            
            // Add tensor as parameter to list
            param_list->append(tensor);
        }
        
        // Test various operations on the ParameterList
        if (param_list->size() > 0) {
            // Access parameters by index
            for (size_t i = 0; i < param_list->size(); ++i) {
                torch::Tensor param = (*param_list)[i];
                
                // Try some operations on the parameter
                if (param.defined() && param.numel() > 0) {
                    torch::Tensor result = param.clone();
                    if (param.is_floating_point()) {
                        result = param * 2.0;
                    }
                }
            }
            
            // Test iteration
            for (const auto& param : param_list->parameters()) {
                if (param.defined()) {
                    torch::Tensor result = param.clone();
                }
            }
            
            // Test append
            if (offset < Size) {
                torch::Tensor new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                param_list->append(new_tensor);
            }
            
            // Test extend
            if (offset < Size) {
                torch::nn::ParameterList second_list;
                uint8_t second_size = (offset < Size) ? (Data[offset++] % 5 + 1) : 1;
                
                for (uint8_t i = 0; i < second_size && offset < Size; ++i) {
                    torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    second_list->append(tensor);
                }
                
                param_list->extend(*second_list);
            }
            
            // Test named parameters
            auto named_params = param_list->named_parameters();
            for (const auto& pair : named_params) {
                const std::string& name = pair.key();
                const torch::Tensor& param = pair.value();
                
                if (param.defined() && param.numel() > 0) {
                    torch::Tensor result = param.clone();
                }
            }
        }
        
        // Test empty ParameterList
        torch::nn::ParameterList empty_list;
        auto empty_params = empty_list->parameters();
        
        // Test with invalid operations
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Try to access out-of-bounds index
                if (param_list->size() > 0) {
                    size_t invalid_idx = param_list->size() + 1;
                    torch::Tensor invalid_param = (*param_list)[invalid_idx];
                }
            } catch (const std::exception&) {
                // Expected exception, continue
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
