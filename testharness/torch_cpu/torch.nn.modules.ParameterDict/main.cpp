#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a ParameterDict
        torch::nn::ParameterDict paramDict;
        
        // Determine number of parameters to add (1-10)
        uint8_t numParams = (Size > 0) ? (Data[offset] % 10) + 1 : 1;
        offset = std::min(offset + 1, Size);
        
        // Add parameters to the ParameterDict
        for (uint8_t i = 0; i < numParams && offset < Size; ++i) {
            // Create a tensor for the parameter
            torch::Tensor paramTensor;
            try {
                paramTensor = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // If tensor creation fails, create a simple tensor
                paramTensor = torch::ones({1, 1});
            }
            
            // Create a key for the parameter
            std::string key = "param" + std::to_string(i);
            
            // Add the parameter to the ParameterDict
            paramDict->insert(key, paramTensor);
        }
        
        // Test various ParameterDict operations
        if (paramDict->size() > 0) {
            // Test contains
            std::string firstKey = "param0";
            bool contains = paramDict->contains(firstKey);
            
            // Test get
            if (contains) {
                torch::Tensor param = paramDict->get(firstKey);
                
                // Test update with another parameter
                torch::Tensor newParam = torch::ones({2, 2});
                paramDict->update(firstKey, newParam);
            }
            
            // Test iteration
            for (const auto& pair : *paramDict) {
                const std::string& name = pair.key();
                const torch::Tensor& param = pair.value();
                
                // Use the parameter to prevent compiler optimization
                if (param.numel() > 0) {
                    torch::Tensor result = param.sum();
                }
            }
            
            // Test remove if we have at least one parameter
            if (paramDict->size() > 0) {
                std::string keyToRemove = "param0";
                if (paramDict->contains(keyToRemove)) {
                    paramDict->remove(keyToRemove);
                }
            }
            
            // Test clear
            paramDict->clear();
            
            // Add parameters again after clearing
            for (uint8_t i = 0; i < numParams && offset < Size; ++i) {
                torch::Tensor paramTensor;
                try {
                    paramTensor = fuzzer_utils::createTensor(Data, Size, offset);
                } catch (const std::exception& e) {
                    paramTensor = torch::ones({1, 1});
                }
                
                std::string key = "new_param" + std::to_string(i);
                paramDict->insert(key, paramTensor);
            }
            
            // Test empty
            bool isEmpty = paramDict->empty();
            
            // Test copy constructor
            torch::nn::ParameterDict paramDictCopy = paramDict;
            
            // Test move constructor
            torch::nn::ParameterDict paramDictMove = std::move(paramDictCopy);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}