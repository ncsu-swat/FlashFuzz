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
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a ParameterDict
        torch::nn::ParameterDict paramDict;
        
        // Determine number of parameters to add (1-10)
        uint8_t numParams = (Data[offset] % 10) + 1;
        offset++;
        
        // Add parameters to the ParameterDict
        for (uint8_t i = 0; i < numParams && offset < Size; i++) {
            torch::Tensor tensor;
            try {
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (...) {
                // If tensor creation fails, use a simple tensor
                tensor = torch::randn({2, 2});
            }
            
            std::string key = "param" + std::to_string(i);
            
            // Insert tensor as parameter
            paramDict->insert(key, tensor);
        }
        
        // Test various ParameterDict operations
        if (paramDict->size() > 0) {
            // Test contains
            std::string firstKey = "param0";
            bool hasKey = paramDict->contains(firstKey);
            
            // Test access via operator[]
            if (hasKey) {
                torch::Tensor param = paramDict[firstKey];
                // Do something with param to ensure it's exercised
                auto shape = param.sizes();
            }
            
            // Test keys()
            auto keys = paramDict->keys();
            
            // Test values()
            auto values = paramDict->values();
            
            // Test iteration via keys and operator[]
            for (const auto& key : paramDict->keys()) {
                torch::Tensor param = paramDict[key];
                // Access to exercise iteration
                auto dtype = param.dtype();
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
            
            // Test clear based on fuzz data
            if (offset < Size && Data[offset] % 2 == 0) {
                paramDict->clear();
            }
            offset++;
        }
        
        // Test empty ParameterDict
        torch::nn::ParameterDict emptyDict;
        
        // Test if empty using size() == 0
        bool isEmpty = (emptyDict->size() == 0);
        
        // Test adding parameters with different dtypes
        if (offset + 1 < Size) {
            emptyDict->insert("float_param", torch::randn({3, 3}, torch::kFloat));
            emptyDict->insert("double_param", torch::randn({2, 2}, torch::kDouble));
        }
        
        // Test update from another ParameterDict
        if (offset < Size && paramDict->size() > 0 && Data[offset] % 2 == 0) {
            try {
                emptyDict->update(*paramDict);
            } catch (...) {
                // Update may fail if keys conflict, ignore
            }
        }
        
        // Test parameters() - returns vector of tensors
        auto params = paramDict->parameters();
        for (const auto& p : params) {
            auto numel = p.numel();
        }
        
        // Test named_parameters() - returns OrderedDict
        auto namedParams = paramDict->named_parameters();
        for (const auto& np : namedParams) {
            const std::string& name = np.key();
            const torch::Tensor& tensor = np.value();
            // Use the variables to avoid warnings
            (void)name;
            (void)tensor;
        }
        
        // Test size()
        size_t dictSize = paramDict->size();
        (void)dictSize;
        (void)isEmpty;
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}