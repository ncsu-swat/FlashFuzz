#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a ParameterDict
        torch::nn::ParameterDict paramDict;
        
        // Determine number of parameters to add (1-10)
        uint8_t numParams = (Size > 0) ? (Data[offset] % 10) + 1 : 1;
        offset++;
        
        // Add parameters to the ParameterDict
        for (uint8_t i = 0; i < numParams && offset < Size; i++) {
            // Create a tensor to use as parameter
            torch::Tensor tensor;
            try {
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // If tensor creation fails, try with a simple tensor
                tensor = torch::ones({1});
            }
            
            // Create a parameter key
            std::string key = "param" + std::to_string(i);
            
            // Add parameter to the dict
            paramDict->insert(key, tensor);
        }
        
        // Test various ParameterDict operations
        if (paramDict->size() > 0) {
            // Test contains
            std::string firstKey = "param0";
            bool contains = paramDict->contains(firstKey);
            
            // Test get
            if (contains) {
                torch::Tensor param = paramDict->get(firstKey);
            }
            
            // Test iteration
            for (const auto& pair : *paramDict) {
                const std::string& name = pair.key();
                const torch::Tensor& param = pair.value();
            }
            
            // Test removal if we have at least one parameter
            if (paramDict->size() > 0 && offset < Size) {
                uint8_t indexToRemove = Data[offset] % paramDict->size();
                offset++;
                std::string keyToRemove = "param" + std::to_string(indexToRemove);
                if (paramDict->contains(keyToRemove)) {
                    paramDict->pop(keyToRemove);
                }
            }
            
            // Test clear
            if (offset < Size && Data[offset] % 2 == 0) {
                paramDict->clear();
            }
            offset++;
        }
        
        // Test empty ParameterDict
        torch::nn::ParameterDict emptyDict;
        
        // Test adding parameters with different dtypes
        if (offset + 1 < Size) {
            emptyDict->insert("float_param", torch::ones({1}, torch::kFloat));
            emptyDict->insert("int_param", torch::ones({1}, torch::kInt));
            emptyDict->insert("bool_param", torch::ones({1}, torch::kBool));
        }
        
        // Test update from another ParameterDict
        if (offset < Size && Data[offset] % 2 == 0) {
            emptyDict->update(*paramDict);
        }
        
        // Test named_parameters
        for (const auto& param : paramDict->named_parameters()) {
            const std::string& name = param.key();
            const torch::Tensor& tensor = param.value();
        }
        
        // Test parameters
        for (const auto& param : paramDict->parameters()) {
            // Access the parameter tensor
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}