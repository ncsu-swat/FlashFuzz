#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Create a ModuleDict
        torch::nn::ModuleDict moduleDict;
        
        // Determine number of modules to add (1-10)
        uint8_t numModules = (Data[offset] % 10) + 1;
        offset++;
        
        // Add modules to the ModuleDict
        for (uint8_t i = 0; i < numModules && offset < Size; i++) {
            // Create a key for the module
            std::string key = "module_" + std::to_string(i);
            
            // Determine module type based on data
            uint8_t moduleType = Data[offset] % 5;
            offset++;
            
            // Create different types of modules based on the data
            switch (moduleType) {
                case 0: {
                    // Linear module
                    int64_t inFeatures = (offset < Size) ? (Data[offset] % 100) + 1 : 10;
                    offset++;
                    int64_t outFeatures = (offset < Size) ? (Data[offset] % 100) + 1 : 5;
                    offset++;
                    auto linear = torch::nn::Linear(inFeatures, outFeatures);
                    moduleDict->insert(key, linear);
                    break;
                }
                case 1: {
                    // Conv2d module
                    int64_t inChannels = (offset < Size) ? (Data[offset] % 16) + 1 : 3;
                    offset++;
                    int64_t outChannels = (offset < Size) ? (Data[offset] % 16) + 1 : 6;
                    offset++;
                    int64_t kernelSize = (offset < Size) ? (Data[offset] % 7) + 1 : 3;
                    offset++;
                    auto conv = torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize));
                    moduleDict->insert(key, conv);
                    break;
                }
                case 2: {
                    // ReLU module
                    bool inplace = (offset < Size) ? (Data[offset] % 2 == 0) : false;
                    offset++;
                    auto relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace));
                    moduleDict->insert(key, relu);
                    break;
                }
                case 3: {
                    // Dropout module
                    double prob = (offset < Size) ? static_cast<double>(Data[offset]) / 255.0 : 0.5;
                    offset++;
                    auto dropout = torch::nn::Dropout(torch::nn::DropoutOptions(prob));
                    moduleDict->insert(key, dropout);
                    break;
                }
                case 4: {
                    // BatchNorm1d module
                    int64_t numFeatures = (offset < Size) ? (Data[offset] % 64) + 1 : 10;
                    offset++;
                    auto bn = torch::nn::BatchNorm1d(numFeatures);
                    moduleDict->insert(key, bn);
                    break;
                }
            }
        }
        
        // Test basic ModuleDict operations
        
        // 1. Test size and empty
        size_t dictSize = moduleDict->size();
        bool isEmpty = moduleDict->empty();
        (void)dictSize;
        (void)isEmpty;
        
        // 2. Test contains
        bool containsFirst = moduleDict->contains("module_0");
        (void)containsFirst;
        
        // 3. Test keys
        auto keys = moduleDict->keys();
        
        // 4. Test values
        auto values = moduleDict->values();
        (void)values;
        
        // 5. Test items
        auto items = moduleDict->items();
        (void)items;
        
        // 6. Test clear based on data
        bool shouldClear = false;
        if (offset < Size) {
            shouldClear = (Data[offset] % 4 == 0); // 25% chance
            offset++;
        }
        
        // 7. Test pop before clear
        if (!moduleDict->empty() && offset < Size && !shouldClear) {
            std::string keyToPop = "module_" + std::to_string(Data[offset] % numModules);
            offset++;
            if (moduleDict->contains(keyToPop)) {
                try {
                    auto poppedModule = moduleDict->pop(keyToPop);
                    (void)poppedModule;
                } catch (...) {
                    // Silent catch for expected failures
                }
            }
        }
        
        // 8. Test update with another ModuleDict
        if (!shouldClear) {
            torch::nn::ModuleDict anotherDict;
            auto extraLinear = torch::nn::Linear(5, 3);
            anotherDict->insert("extra_linear", extraLinear);
            moduleDict->update(*anotherDict);
        }
        
        // 9. Now perform clear if selected
        if (shouldClear) {
            moduleDict->clear();
        }
        
        // 10. Test iteration over modules
        if (!moduleDict->empty()) {
            for (const auto& item : moduleDict->items()) {
                auto key = item.key();
                auto module = item.value();
                (void)key;
                (void)module;
            }
        }
        
        // 11. Test forward pass through modules
        if (!moduleDict->empty()) {
            torch::NoGradGuard no_grad;
            
            for (const auto& key : moduleDict->keys()) {
                try {
                    auto module = moduleDict[key];
                    
                    // Try different forward patterns based on module type
                    torch::Tensor inputTensor;
                    torch::Tensor output;
                    
                    // Create appropriate input based on fuzzer data
                    if (offset + 2 < Size) {
                        int64_t dim1 = (Data[offset] % 32) + 1;
                        int64_t dim2 = (Data[offset + 1] % 64) + 1;
                        offset += 2;
                        
                        // Try as Linear input
                        try {
                            inputTensor = torch::randn({dim1, dim2});
                            output = module->as<torch::nn::Linear>()->forward(inputTensor);
                        } catch (...) {}
                        
                        // Try as Conv2d input
                        try {
                            int64_t channels = (Data[offset % Size] % 16) + 1;
                            inputTensor = torch::randn({1, channels, 28, 28});
                            output = module->as<torch::nn::Conv2d>()->forward(inputTensor);
                        } catch (...) {}
                        
                        // Try as ReLU/Dropout input
                        try {
                            inputTensor = torch::randn({dim1, dim2});
                            output = module->as<torch::nn::ReLU>()->forward(inputTensor);
                        } catch (...) {}
                        
                        try {
                            inputTensor = torch::randn({dim1, dim2});
                            output = module->as<torch::nn::Dropout>()->forward(inputTensor);
                        } catch (...) {}
                        
                        // Try as BatchNorm1d input
                        try {
                            int64_t features = (Data[offset % Size] % 64) + 1;
                            inputTensor = torch::randn({dim1, features});
                            output = module->as<torch::nn::BatchNorm1d>()->forward(inputTensor);
                        } catch (...) {}
                        
                        (void)output;
                    }
                } catch (...) {
                    // Silent catch for expected failures during forward
                }
            }
        }
        
        // 12. Test re-adding modules after operations
        if (offset < Size && moduleDict->size() < 5) {
            std::string newKey = "late_module_" + std::to_string(Data[offset] % 100);
            offset++;
            int64_t features = (offset < Size) ? (Data[offset] % 50) + 1 : 10;
            auto lateLinear = torch::nn::Linear(features, features);
            moduleDict->insert(newKey, lateLinear);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}