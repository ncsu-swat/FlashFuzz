#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/torch.h>

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
        
        // Ensure we have enough data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a ModuleList
        torch::nn::ModuleList moduleList;
        
        // Determine number of modules to add (1-10)
        uint8_t numModules = (Data[offset] % 10) + 1;
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
        size_t listSize = moduleList->size();
        
        // 2. Test is_empty() method
        bool isEmpty = moduleList->is_empty();
        
        // 3. Test iteration through modules using begin/end
        for (auto it = moduleList->begin(); it != moduleList->end(); ++it) {
            auto paramSize = (*it)->parameters().size();
            (void)paramSize; // suppress unused variable warning
        }
        
        // 4. Test indexing with at() and operator[]
        if (listSize > 0) {
            auto firstModule = moduleList[0];
            
            // Test ptr() method for accessing underlying module
            auto modulePtr = moduleList->ptr(0);
            (void)modulePtr;
        }
        
        // 5. Test extend method if we have modules
        if (listSize > 0) {
            torch::nn::ModuleList additionalModules;
            additionalModules->push_back(torch::nn::Linear(10, 5));
            additionalModules->push_back(torch::nn::ReLU());
            
            moduleList->extend(*additionalModules);
        }
        
        // 6. Test push_back method with different module
        moduleList->push_back(torch::nn::Linear(5, 1));
        
        // 7. Test children() method
        auto children = moduleList->children();
        (void)children;
        
        // 8. Test named_children()
        auto namedChildren = moduleList->named_children();
        (void)namedChildren;
        
        // 9. Test parameters() method
        auto params = moduleList->parameters();
        (void)params;
        
        // 10. Test named_parameters()
        auto namedParams = moduleList->named_parameters();
        (void)namedParams;
        
        // 11. Create a tensor and try to pass it through compatible modules
        if (offset < Size && Size - offset > 2) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Only process the tensor if it's valid
                if (tensor.defined() && tensor.numel() > 0) {
                    // Try to pass the tensor through each module
                    for (size_t i = 0; i < moduleList->size(); ++i) {
                        auto module = moduleList->ptr(i);
                        
                        // Check if the module is a Linear layer
                        if (auto linearPtr = std::dynamic_pointer_cast<torch::nn::LinearImpl>(module)) {
                            // Reshape tensor to match Linear layer input requirements
                            if (tensor.dim() >= 1) {
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
                            // Conv2d requires 4D input (batch, channels, height, width)
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
                // Catch exceptions from tensor operations silently - these are expected
                // when tensor shapes don't match module requirements
            }
        }
        
        // 12. Test train/eval mode switching
        moduleList->train();
        moduleList->eval();
        
        // 13. Test to() method for device/dtype conversion
        moduleList->to(torch::kFloat32);
        
        // 14. Test zero_grad()
        moduleList->zero_grad();
        
        // 15. Test clone() if available
        try {
            auto cloned = moduleList->clone();
            (void)cloned;
        } catch (...) {
            // clone may not be available for all configurations
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}