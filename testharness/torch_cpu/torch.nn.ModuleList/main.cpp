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
        
        // Ensure we have enough data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a ModuleList
        torch::nn::ModuleList moduleList;
        
        // Determine number of modules to add (1-10)
        uint8_t numModules = (Size > 0) ? (Data[offset] % 10) + 1 : 3;
        offset++;
        
        // Create and add modules to the ModuleList
        for (uint8_t i = 0; i < numModules && offset < Size; i++) {
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
                    
                    auto linear = torch::nn::Linear(inFeatures, outFeatures);
                    moduleList->push_back(linear);
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
                        torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize)
                    );
                    moduleList->push_back(conv);
                    break;
                }
                case 2: {
                    // ReLU module
                    bool inplace = (offset < Size) ? (Data[offset] % 2 == 0) : false;
                    offset++;
                    
                    auto relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace));
                    moduleList->push_back(relu);
                    break;
                }
                case 3: {
                    // Dropout module
                    double prob = (offset < Size) ? static_cast<double>(Data[offset]) / 255.0 : 0.5;
                    offset++;
                    
                    auto dropout = torch::nn::Dropout(torch::nn::DropoutOptions(prob));
                    moduleList->push_back(dropout);
                    break;
                }
                case 4: {
                    // BatchNorm2d module
                    int64_t numFeatures = (offset < Size) ? (Data[offset] % 32) + 1 : 10;
                    offset++;
                    
                    auto batchnorm = torch::nn::BatchNorm2d(numFeatures);
                    moduleList->push_back(batchnorm);
                    break;
                }
            }
        }
        
        // Test ModuleList operations
        
        // 1. Test size() method
        auto size = moduleList->size();
        
        // 2. Test iteration through modules
        for (const auto& module : *moduleList) {
            auto paramSize = module->parameters().size();
        }
        
        // 3. Test indexed access
        if (moduleList->size() > 0) {
            auto firstModule = (*moduleList)[0];
            auto lastModule = (*moduleList)[moduleList->size() - 1];
        }
        
        // 4. Test extend method if we have enough modules
        if (moduleList->size() >= 2) {
            torch::nn::ModuleList otherList;
            
            // Add a couple of modules to the other list
            otherList->push_back(torch::nn::Linear(10, 5));
            otherList->push_back(torch::nn::ReLU());
            
            // Extend the original list with the other list
            moduleList->extend(*otherList);
        }
        
        // 5. Test push_back method instead of append
        moduleList->push_back(torch::nn::Linear(20, 10));
        
        // 6. Test insert method
        if (moduleList->size() > 0) {
            size_t insertIdx = (offset < Size) ? Data[offset] % moduleList->size() : 0;
            offset++;
            moduleList->insert(insertIdx, torch::nn::Dropout(0.2));
        }
        
        // 7. Create a tensor and pass it through modules that can handle it
        if (offset < Size) {
            try {
                // Create a tensor with appropriate dimensions for the first module
                torch::Tensor input;
                
                // Try to determine what kind of input the first module expects
                if (moduleList->size() > 0) {
                    auto firstModule = (*moduleList)[0];
                    
                    // Check module type and create appropriate input
                    if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(firstModule)) {
                        int64_t batchSize = 2;
                        int64_t inFeatures = linear->options.in_features();
                        input = torch::rand({batchSize, inFeatures});
                        auto output = linear->forward(input);
                    }
                    else if (auto conv = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(firstModule)) {
                        int64_t batchSize = 2;
                        int64_t inChannels = conv->options.in_channels();
                        int64_t height = 28;
                        int64_t width = 28;
                        input = torch::rand({batchSize, inChannels, height, width});
                        auto output = conv->forward(input);
                    }
                    else if (auto relu = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(firstModule)) {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        auto output = relu->forward(input);
                    }
                    else if (auto dropout = std::dynamic_pointer_cast<torch::nn::DropoutImpl>(firstModule)) {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        auto output = dropout->forward(input);
                    }
                    else if (auto batchnorm = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(firstModule)) {
                        int64_t batchSize = 2;
                        int64_t numFeatures = batchnorm->options.num_features();
                        int64_t height = 28;
                        int64_t width = 28;
                        input = torch::rand({batchSize, numFeatures, height, width});
                        auto output = batchnorm->forward(input);
                    }
                }
            }
            catch (const std::exception& e) {
                // Catch exceptions from tensor creation or forward pass
            }
        }
        
        // 8. Test children() method
        auto children = moduleList->children();
        for (const auto& child : children) {
            auto paramSize = child->parameters().size();
        }
        
        // 9. Test named_children() method
        auto namedChildren = moduleList->named_children();
        for (const auto& item : namedChildren) {
            auto name = item.key();
            auto child = item.value();
            auto paramSize = child->parameters().size();
        }
        
        // 10. Test pop_back method instead of clear
        if (offset < Size && Data[offset] % 2 == 0 && moduleList->size() > 0) {
            moduleList->pop_back();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
