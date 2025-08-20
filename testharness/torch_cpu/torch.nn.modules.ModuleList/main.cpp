#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Create a ModuleList
        torch::nn::ModuleList moduleList;
        
        // Determine number of modules to add (1-10)
        uint8_t numModules = (Size > 0) ? (Data[offset] % 10) + 1 : 3;
        offset = std::min(offset + 1, Size);
        
        // Create and add modules to the ModuleList
        for (uint8_t i = 0; i < numModules && offset < Size; ++i) {
            // Determine module type based on input data
            uint8_t moduleType = (offset < Size) ? Data[offset] % 5 : 0;
            offset = std::min(offset + 1, Size);
            
            // Create different types of modules based on the moduleType
            switch (moduleType) {
                case 0: {
                    // Linear module
                    int64_t inFeatures = (offset < Size) ? (Data[offset] % 100) + 1 : 10;
                    offset = std::min(offset + 1, Size);
                    int64_t outFeatures = (offset < Size) ? (Data[offset] % 100) + 1 : 5;
                    offset = std::min(offset + 1, Size);
                    
                    moduleList->push_back(torch::nn::Linear(inFeatures, outFeatures));
                    break;
                }
                case 1: {
                    // Conv2d module
                    int64_t inChannels = (offset < Size) ? (Data[offset] % 16) + 1 : 3;
                    offset = std::min(offset + 1, Size);
                    int64_t outChannels = (offset < Size) ? (Data[offset] % 16) + 1 : 6;
                    offset = std::min(offset + 1, Size);
                    int64_t kernelSize = (offset < Size) ? (Data[offset] % 5) + 1 : 3;
                    offset = std::min(offset + 1, Size);
                    
                    moduleList->push_back(torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize)
                    ));
                    break;
                }
                case 2: {
                    // ReLU module
                    bool inplace = (offset < Size) ? (Data[offset] % 2 == 0) : false;
                    offset = std::min(offset + 1, Size);
                    
                    moduleList->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace)));
                    break;
                }
                case 3: {
                    // Dropout module
                    double prob = (offset < Size) ? static_cast<double>(Data[offset]) / 255.0 : 0.5;
                    offset = std::min(offset + 1, Size);
                    
                    moduleList->push_back(torch::nn::Dropout(torch::nn::DropoutOptions(prob)));
                    break;
                }
                case 4: {
                    // BatchNorm2d module
                    int64_t numFeatures = (offset < Size) ? (Data[offset] % 32) + 1 : 10;
                    offset = std::min(offset + 1, Size);
                    
                    moduleList->push_back(torch::nn::BatchNorm2d(numFeatures));
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
        
        // 3. Test indexing
        if (moduleList->size() > 0) {
            auto firstModule = (*moduleList)[0];
        }
        
        // 4. Test extend method if we have enough modules
        if (numModules > 2) {
            torch::nn::ModuleList additionalModules;
            additionalModules->push_back(torch::nn::Linear(10, 5));
            additionalModules->push_back(torch::nn::ReLU());
            
            moduleList->extend(*additionalModules);
        }
        
        // 5. Test push_back method instead of append
        moduleList->push_back(torch::nn::Linear(5, 1));
        
        // 6. Create a tensor and try to pass it through the modules
        if (offset < Size && Size - offset > 2) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Only process the tensor if it's valid for the modules
                if (tensor.defined() && tensor.numel() > 0) {
                    // Try to pass the tensor through each module
                    for (size_t i = 0; i < moduleList->size(); ++i) {
                        auto module = (*moduleList)[i];
                        
                        // Check if the module is a Linear layer
                        if (auto linearPtr = std::dynamic_pointer_cast<torch::nn::LinearImpl>(module)) {
                            // Reshape tensor to match Linear layer input requirements
                            if (tensor.dim() > 1) {
                                int64_t inFeatures = linearPtr->options.in_features();
                                if (tensor.size(-1) == inFeatures) {
                                    tensor = linearPtr->forward(tensor);
                                }
                            }
                        }
                        // Check if the module is a ReLU
                        else if (auto reluPtr = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(module)) {
                            tensor = reluPtr->forward(tensor);
                        }
                        // Check if the module is a Dropout
                        else if (auto dropoutPtr = std::dynamic_pointer_cast<torch::nn::DropoutImpl>(module)) {
                            tensor = dropoutPtr->forward(tensor);
                        }
                        // Check if the module is Conv2d
                        else if (auto convPtr = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(module)) {
                            // Reshape tensor to match Conv2d input requirements if needed
                            if (tensor.dim() == 4) {
                                int64_t inChannels = convPtr->options.in_channels();
                                if (tensor.size(1) == inChannels) {
                                    tensor = convPtr->forward(tensor);
                                }
                            }
                        }
                        // Check if the module is BatchNorm2d
                        else if (auto bnPtr = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(module)) {
                            if (tensor.dim() == 4) {
                                int64_t numFeatures = bnPtr->options.num_features();
                                if (tensor.size(1) == numFeatures) {
                                    tensor = bnPtr->forward(tensor);
                                }
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Catch exceptions from tensor operations but continue fuzzing
            }
        }
        
        // 7. Test clear method
        moduleList->clear();
        
        // 8. Test insert method
        moduleList->insert(0, torch::nn::Linear(10, 5));
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}