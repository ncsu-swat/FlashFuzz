#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
            
            // Add the parameter to the ParameterDict using insert
            paramDict->insert(key, paramTensor);
        }
        
        // Test various ParameterDict operations
        if (paramDict->size() > 0) {
            // Test contains
            std::string firstKey = "param0";
            bool contains = paramDict->contains(firstKey);
            
            // Test operator[] access
            if (contains) {
                torch::Tensor param = paramDict[firstKey];
                
                // Use the parameter to prevent compiler optimization
                if (param.numel() > 0) {
                    torch::Tensor result = param.sum();
                    (void)result;
                }
            }
            
            // Test keys() method
            std::vector<std::string> keys = paramDict->keys();
            for (const auto& key : keys) {
                torch::Tensor param = paramDict[key];
                if (param.numel() > 0) {
                    torch::Tensor result = param.mean();
                    (void)result;
                }
            }
            
            // Test values() method
            std::vector<torch::Tensor> values = paramDict->values();
            for (const auto& val : values) {
                if (val.numel() > 0) {
                    torch::Tensor result = val.abs();
                    (void)result;
                }
            }
            
            // Test pop if we have at least one parameter
            if (paramDict->contains(firstKey)) {
                torch::Tensor popped = paramDict->pop(firstKey);
                (void)popped;
            }
            
            // Test clear
            paramDict->clear();
            
            // Verify empty after clear
            bool isEmpty = paramDict->empty();
            (void)isEmpty;
            
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
            
            // Test size after re-adding
            size_t newSize = paramDict->size();
            (void)newSize;
            
            // Iterate over the ParameterDict using keys
            std::vector<std::string> allKeys = paramDict->keys();
            for (const auto& key : allKeys) {
                torch::Tensor param = paramDict[key];
                if (param.defined() && param.numel() > 0) {
                    torch::Tensor result = param.clone();
                    (void)result;
                }
            }
        }
        
        // Test creating a second ParameterDict and using update
        torch::nn::ParameterDict paramDict2;
        paramDict2->insert("extra_param", torch::zeros({3, 3}));
        
        // Update paramDict with paramDict2
        try {
            paramDict->update(*paramDict2);
        } catch (...) {
            // update may fail, silently ignore
        }
        
        // Verify the update worked by checking for the extra_param key
        if (paramDict->contains("extra_param")) {
            torch::Tensor extraParam = paramDict["extra_param"];
            if (extraParam.defined()) {
                torch::Tensor result = extraParam.sum();
                (void)result;
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