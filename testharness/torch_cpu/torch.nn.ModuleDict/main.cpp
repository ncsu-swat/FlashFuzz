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
        
        // Skip if not enough data
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
            
            // Determine module type based on data
            uint8_t moduleType = (offset < Size) ? Data[offset] % 5 : 0;
            offset++;
            
            // Create different types of modules based on the data
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
                    auto seq = torch::nn::Sequential(
                        torch::nn::Linear(10, 5),
                        torch::nn::ReLU(),
                        torch::nn::Linear(5, 1)
                    );
                    moduleDict[key] = seq;
                    break;
                }
            }
        }
        
        // Test basic ModuleDict operations
        
        // 1. Test size and empty
        size_t dictSize = moduleDict->size();
        bool isEmpty = moduleDict->empty();
        
        // 2. Test contains
        bool containsFirst = moduleDict->contains("module_0");
        
        // 3. Test keys
        auto keys = moduleDict->keys();
        
        // 4. Test values
        auto values = moduleDict->values();
        
        // 5. Test items
        auto items = moduleDict->items();
        
        // 6. Test clear
        if (offset < Size && Data[offset] % 2 == 0) {
            moduleDict->clear();
        }
        offset++;
        
        // 7. Test pop
        if (!moduleDict->empty() && offset < Size) {
            std::string keyToPop = "module_" + std::to_string(Data[offset] % numModules);
            if (moduleDict->contains(keyToPop)) {
                auto poppedModule = moduleDict->pop(keyToPop);
            }
        }
        offset++;
        
        // 8. Test update with another ModuleDict
        torch::nn::ModuleDict anotherDict;
        anotherDict["extra_module"] = torch::nn::Linear(5, 3);
        moduleDict->update(anotherDict);
        
        // 9. Test forward pass with a tensor if we have any modules
        if (!moduleDict->empty()) {
            // Create a test tensor
            torch::Tensor testTensor;
            try {
                testTensor = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // If tensor creation fails, create a simple one
                testTensor = torch::ones({10, 10});
            }
            
            // Try to forward through a module if it exists
            for (const auto& key : moduleDict->keys()) {
                try {
                    // Reshape tensor if needed for the specific module
                    torch::Tensor inputTensor = testTensor;
                    auto& module = moduleDict->at<torch::nn::AnyModule>(key);
                    
                    // Adapt tensor shape based on module type
                    if (auto linear = module.try_get<torch::nn::Linear>()) {
                        // For Linear, ensure last dimension matches in_features
                        int64_t inFeatures = linear->options.in_features();
                        if (testTensor.dim() == 0) {
                            inputTensor = torch::ones({inFeatures});
                        } else if (testTensor.dim() == 1 && testTensor.size(0) != inFeatures) {
                            inputTensor = torch::ones({inFeatures});
                        } else if (testTensor.dim() > 1 && testTensor.size(-1) != inFeatures) {
                            std::vector<int64_t> newShape(testTensor.dim());
                            for (int i = 0; i < testTensor.dim() - 1; i++) {
                                newShape[i] = testTensor.size(i);
                            }
                            newShape[testTensor.dim() - 1] = inFeatures;
                            inputTensor = torch::ones(newShape);
                        }
                        auto output = module.forward(inputTensor);
                    } else if (auto conv = module.try_get<torch::nn::Conv2d>()) {
                        // For Conv2d, ensure NCHW format with matching channels
                        int64_t inChannels = conv->options.in_channels();
                        inputTensor = torch::ones({1, inChannels, 28, 28});
                        auto output = module.forward(inputTensor);
                    } else if (module.try_get<torch::nn::ReLU>() || 
                               module.try_get<torch::nn::Dropout>()) {
                        // For ReLU and Dropout, any tensor shape works
                        if (testTensor.dim() == 0) {
                            inputTensor = torch::ones({1, 1});
                        }
                        auto output = module.forward(inputTensor);
                    } else {
                        // For other modules, try with the original tensor
                        // This might throw if incompatible
                        auto output = module.forward(inputTensor);
                    }
                } catch (const std::exception& e) {
                    // Ignore exceptions from forward pass - they're expected for incompatible shapes
                }
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
