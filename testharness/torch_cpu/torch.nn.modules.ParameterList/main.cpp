#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Ensure we have enough data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a ParameterList
        torch::nn::ParameterList paramList;
        
        // Determine how many parameters to add to the list
        uint8_t numParams = Data[offset++] % 10 + 1; // 1-10 parameters
        
        // Create and add parameters to the list
        for (uint8_t i = 0; i < numParams && offset < Size; ++i) {
            // Create a tensor to use as parameter
            torch::Tensor tensor;
            try {
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // If tensor creation fails, create a simple tensor
                tensor = torch::ones({1});
            }
            
            // Add the tensor as a parameter to the list
            tensor.requires_grad_(true);
            paramList->append(tensor);
        }
        
        // Test various operations on the ParameterList
        if (paramList->size() > 0) {
            // Access parameters by index
            auto firstParam = (*paramList)[0];
            
            // Test iteration
            for (const auto& param : paramList->named_parameters()) {
                auto shape = param.value().sizes();
            }
            
            // Test extend functionality if we have enough parameters
            if (paramList->size() >= 2) {
                torch::nn::ParameterList secondList;
                torch::Tensor tensor2 = torch::ones({2, 2});
                tensor2.requires_grad_(true);
                secondList->append(tensor2);
                
                // Extend the first list with the second
                paramList->extend(*secondList);
            }
            
            // Test append functionality
            torch::Tensor tensor3 = torch::zeros({3, 3});
            tensor3.requires_grad_(true);
            paramList->append(tensor3);
            
            // Test named_parameters
            auto namedParams = paramList->named_parameters();
            for (const auto& pair : namedParams) {
                auto name = pair.key();
                auto param = pair.value();
            }
            
            // Test parameters
            auto params = paramList->parameters();
            
            // Test to(device) functionality
            paramList->to(torch::kCPU);
            
            // Test to(dtype) functionality
            if (offset < Size) {
                uint8_t dtypeSelector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtypeSelector);
                paramList->to(dtype);
            }
        }
        
        // Test empty ParameterList
        torch::nn::ParameterList emptyList;
        auto emptyParams = emptyList->parameters();
        
        // Test ParameterList with a single large tensor
        if (offset < Size) {
            try {
                torch::Tensor largeTensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::nn::ParameterList singleParamList;
                largeTensor.requires_grad_(true);
                singleParamList->append(largeTensor);
            } catch (const std::exception& e) {
                // Ignore exceptions from tensor creation
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