#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/torch.h>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a ModuleDict
        torch::nn::ModuleDict moduleDict;
        
        // Determine number of modules to add (1-10)
        uint8_t numModules = (Data[offset] % 10) + 1;
        offset++;
        
        // Track which module_0 type was created for later forward pass
        int module0Type = -1;
        int64_t module0InFeatures = 0;
        int64_t module0InChannels = 0;
        
        // Add modules to the ModuleDict using update with initializer list
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
                    moduleDict->update({{key, torch::nn::Linear(inFeatures, outFeatures)}});
                    if (i == 0) {
                        module0Type = 0;
                        module0InFeatures = inFeatures;
                    }
                    break;
                }
                case 1: {
                    // Conv2d module
                    int64_t inChannels = (offset < Size) ? (Data[offset] % 16) + 1 : 3;
                    offset++;
                    int64_t outChannels = (offset < Size) ? (Data[offset] % 16) + 1 : 6;
                    offset++;
                    int64_t kernelSize = (offset < Size) ? (Data[offset] % 5) + 1 : 3;
                    offset++;
                    moduleDict->update({{key, torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize))}});
                    if (i == 0) {
                        module0Type = 1;
                        module0InChannels = inChannels;
                    }
                    break;
                }
                case 2: {
                    // ReLU module
                    bool inplace = (offset < Size) ? (Data[offset] % 2 == 0) : false;
                    offset++;
                    moduleDict->update({{key, torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace))}});
                    if (i == 0) {
                        module0Type = 2;
                    }
                    break;
                }
                case 3: {
                    // Dropout module
                    double prob = (offset < Size) ? static_cast<double>(Data[offset]) / 255.0 : 0.5;
                    offset++;
                    moduleDict->update({{key, torch::nn::Dropout(torch::nn::DropoutOptions(prob))}});
                    if (i == 0) {
                        module0Type = 3;
                    }
                    break;
                }
                case 4: {
                    // Sequential module with a few layers
                    moduleDict->update({{key, torch::nn::Sequential(
                        torch::nn::Linear(10, 5),
                        torch::nn::ReLU(),
                        torch::nn::Linear(5, 1)
                    )}});
                    if (i == 0) {
                        module0Type = 4;
                    }
                    break;
                }
            }
        }
        
        // Test various ModuleDict operations
        
        // 1. Test contains method
        if (moduleDict->size() > 0) {
            std::string testKey = "module_0";
            bool contains = moduleDict->contains(testKey);
            (void)contains;
        }
        
        // 2. Test keys method
        auto keys = moduleDict->keys();
        
        // 3. Test values method
        auto values = moduleDict->values();
        
        // 4. Test items method
        auto items = moduleDict->items();
        
        // 5. Test size method
        size_t dictSize = moduleDict->size();
        (void)dictSize;
        
        // 6. Test empty method
        bool isEmpty = moduleDict->is_empty();
        (void)isEmpty;
        
        // 7. Test forward pass based on known module type
        if (moduleDict->contains("module_0") && module0Type >= 0) {
            try {
                torch::NoGradGuard no_grad;
                
                if (module0Type == 0) {
                    // Linear module
                    auto input = torch::rand({2, module0InFeatures});
                    auto module = moduleDict["module_0"]->as<torch::nn::Linear>();
                    if (module) {
                        auto output = module->forward(input);
                    }
                } else if (module0Type == 1) {
                    // Conv2d module
                    auto input = torch::rand({1, module0InChannels, 8, 8});
                    auto module = moduleDict["module_0"]->as<torch::nn::Conv2d>();
                    if (module) {
                        auto output = module->forward(input);
                    }
                } else if (module0Type == 2) {
                    // ReLU module
                    auto input = torch::rand({2, 5});
                    auto module = moduleDict["module_0"]->as<torch::nn::ReLU>();
                    if (module) {
                        auto output = module->forward(input);
                    }
                } else if (module0Type == 3) {
                    // Dropout module
                    auto input = torch::rand({2, 5});
                    auto module = moduleDict["module_0"]->as<torch::nn::Dropout>();
                    if (module) {
                        auto output = module->forward(input);
                    }
                } else if (module0Type == 4) {
                    // Sequential module
                    auto input = torch::rand({2, 10});
                    auto module = moduleDict["module_0"]->as<torch::nn::Sequential>();
                    if (module) {
                        auto output = module->forward(input);
                    }
                }
            } catch (...) {
                // Expected - forward might fail for various reasons
            }
        }
        
        // 8. Test operator[] access
        if (moduleDict->size() > 0) {
            std::string accessKey = "module_0";
            if (moduleDict->contains(accessKey)) {
                auto module = moduleDict[accessKey];
            }
        }
        
        // 9. Test iteration through ModuleDict
        for (const auto& item : moduleDict->named_children()) {
            const auto& name = item.key();
            const auto& module = item.value();
            (void)name;
            (void)module;
        }
        
        // 10. Test pop method (with probability)
        if (offset < Size && moduleDict->size() > 0 && Data[offset] % 5 == 0) {
            auto keysList = moduleDict->keys();
            if (!keysList.empty()) {
                std::string popKey = keysList[Data[offset] % keysList.size()];
                auto module = moduleDict->pop(popKey);
            }
        }
        offset++;
        
        // 11. Test update method (with probability)
        if (offset < Size && Data[offset] % 3 == 0) {
            torch::nn::ModuleDict newModules;
            newModules->update({{"new_module", torch::nn::Linear(5, 3)}});
            moduleDict->update(*newModules);
        }
        offset++;
        
        // 12. Test clear method (with probability) - do this last
        if (offset < Size && Data[offset] % 10 == 0) {
            moduleDict->clear();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}