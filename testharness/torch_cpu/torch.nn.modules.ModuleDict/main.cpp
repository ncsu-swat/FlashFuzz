#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a ModuleDict
        torch::nn::ModuleDict moduleDict;
        
        // Determine number of modules to add (1-10)
        uint8_t numModules = (Size > 0) ? (Data[offset] % 10) + 1 : 1;
        offset++;
        
        // Add modules to the ModuleDict
        for (uint8_t i = 0; i < numModules && offset < Size; i++) {
            // Create a key for the module
            std::string key = "module_" + std::to_string(i);
            
            // Determine module type based on input data
            uint8_t moduleType = (offset < Size) ? Data[offset] % 5 : 0;
            offset++;
            
            // Create different types of modules based on the input
            switch (moduleType) {
                case 0: {
                    // Linear module
                    int64_t inFeatures = (offset < Size) ? (Data[offset] % 100) + 1 : 10;
                    offset++;
                    int64_t outFeatures = (offset < Size) ? (Data[offset] % 100) + 1 : 5;
                    offset++;
                    moduleDict[key] = torch::nn::Linear(inFeatures, outFeatures);
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
                    moduleDict[key] = torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize));
                    break;
                }
                case 2: {
                    // ReLU module
                    bool inplace = (offset < Size) ? (Data[offset] % 2 == 0) : false;
                    offset++;
                    moduleDict[key] = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace));
                    break;
                }
                case 3: {
                    // Dropout module
                    double prob = (offset < Size) ? static_cast<double>(Data[offset]) / 255.0 : 0.5;
                    offset++;
                    moduleDict[key] = torch::nn::Dropout(torch::nn::DropoutOptions(prob));
                    break;
                }
                case 4: {
                    // Sequential module with a few layers
                    moduleDict[key] = torch::nn::Sequential(
                        torch::nn::Linear(10, 5),
                        torch::nn::ReLU(),
                        torch::nn::Linear(5, 1)
                    );
                    break;
                }
            }
        }
        
        // Test various ModuleDict operations
        
        // 1. Test contains method
        if (moduleDict->size() > 0) {
            std::string testKey = "module_0";
            bool contains = moduleDict->contains(testKey);
        }
        
        // 2. Test keys method
        auto keys = moduleDict->keys();
        
        // 3. Test values method
        auto values = moduleDict->values();
        
        // 4. Test items method
        auto items = moduleDict->items();
        
        // 5. Test clear method (with probability)
        if (offset < Size && Data[offset] % 10 == 0) {
            moduleDict->clear();
        }
        offset++;
        
        // 6. Test pop method (with probability)
        if (offset < Size && moduleDict->size() > 0 && Data[offset] % 5 == 0) {
            std::string popKey = "module_" + std::to_string(Data[offset] % moduleDict->size());
            if (moduleDict->contains(popKey)) {
                auto module = moduleDict->pop(popKey);
            }
        }
        offset++;
        
        // 7. Test update method (with probability)
        if (offset < Size && Data[offset] % 3 == 0) {
            torch::nn::ModuleDict newModules;
            newModules["new_module"] = torch::nn::Linear(5, 3);
            moduleDict->update(newModules);
        }
        offset++;
        
        // 8. Test forward pass with a tensor if we have a Linear module
        if (moduleDict->contains("module_0")) {
            try {
                auto module = moduleDict->at<torch::nn::LinearImpl>("module_0");
                // Create input tensor for the linear layer
                int64_t batchSize = 2;
                int64_t inFeatures = module.options.in_features();
                auto input = torch::rand({batchSize, inFeatures});
                auto output = module.forward(input);
            } catch (const std::exception& e) {
                // Catch exceptions from forward pass but continue testing
            }
        }
        
        // 9. Test operator[] access
        if (moduleDict->size() > 0) {
            std::string accessKey = "module_0";
            if (moduleDict->contains(accessKey)) {
                auto module = moduleDict[accessKey];
            }
        }
        
        // 10. Test iteration through ModuleDict
        for (const auto& item : moduleDict->named_children()) {
            const auto& name = item.key();
            const auto& module = item.value();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}