#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Ensure we have enough data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a ModuleList
        torch::nn::ModuleList moduleList;
        
        // Determine number of modules to add (1-10)
        uint8_t numModules = (Data[offset] % 10) + 1;
        offset++;
        
        // Create and add modules to the ModuleList
        for (uint8_t i = 0; i < numModules && offset < Size; i++) {
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
        
        // 2. Test is_empty() method
        bool isEmpty = moduleList->is_empty();
        (void)isEmpty;
        
        // 3. Test iteration through modules
        for (const auto& module : *moduleList) {
            auto paramSize = module->parameters().size();
            (void)paramSize; // Suppress unused variable warning
        }
        
        // 4. Test indexed access with operator[]
        if (moduleList->size() > 0) {
            auto firstModule = moduleList[0];
            auto lastModule = moduleList[moduleList->size() - 1];
            (void)firstModule;
            (void)lastModule;
        }
        
        // 5. Test ptr() method for indexed access
        if (moduleList->size() > 0) {
            auto modulePtr = moduleList->ptr(0);
            (void)modulePtr;
        }
        
        // 6. Test extend method with another ModuleList
        // extend expects the modules to be iterated, so we need to pass the dereferenced impl
        if (moduleList->size() >= 2) {
            torch::nn::ModuleList otherList;
            
            // Add a couple of modules to the other list
            otherList->push_back(torch::nn::Linear(10, 5));
            otherList->push_back(torch::nn::ReLU());
            
            // Extend the original list with the other list
            // Pass dereferenced ModuleList (the impl)
            moduleList->extend(*otherList);
        }
        
        // 7. Test push_back method
        moduleList->push_back(torch::nn::Linear(20, 10));
        
        // 8. Test insert method
        if (moduleList->size() > 0 && offset < Size) {
            size_t insertIdx = Data[offset] % moduleList->size();
            offset++;
            moduleList->insert(insertIdx, torch::nn::Dropout(0.2));
        }
        
        // 9. Create a tensor and pass it through modules that can handle it
        if (offset < Size && moduleList->size() > 0) {
            try {
                // Create a tensor with appropriate dimensions for the first module
                torch::Tensor input;
                
                auto firstModule = moduleList->ptr(0);
                
                // Check module type and create appropriate input
                if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(firstModule)) {
                    int64_t batchSize = 2;
                    int64_t inFeatures = linear->options.in_features();
                    input = torch::rand({batchSize, inFeatures});
                    auto output = linear->forward(input);
                    (void)output;
                }
                else if (auto conv = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(firstModule)) {
                    int64_t batchSize = 2;
                    int64_t inChannels = conv->options.in_channels();
                    int64_t height = 28;
                    int64_t width = 28;
                    input = torch::rand({batchSize, inChannels, height, width});
                    auto output = conv->forward(input);
                    (void)output;
                }
                else if (auto relu = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(firstModule)) {
                    input = fuzzer_utils::createTensor(Data, Size, offset);
                    auto output = relu->forward(input);
                    (void)output;
                }
                else if (auto dropout = std::dynamic_pointer_cast<torch::nn::DropoutImpl>(firstModule)) {
                    input = fuzzer_utils::createTensor(Data, Size, offset);
                    auto output = dropout->forward(input);
                    (void)output;
                }
                else if (auto batchnorm = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(firstModule)) {
                    int64_t batchSize = 2;
                    int64_t numFeatures = batchnorm->options.num_features();
                    int64_t height = 28;
                    int64_t width = 28;
                    input = torch::rand({batchSize, numFeatures, height, width});
                    auto output = batchnorm->forward(input);
                    (void)output;
                }
            }
            catch (const std::exception&) {
                // Silently catch exceptions from tensor creation or forward pass
                // These are expected for invalid shape combinations
            }
        }
        
        // 10. Test children() method
        auto children = moduleList->children();
        for (const auto& child : children) {
            auto paramSize = child->parameters().size();
            (void)paramSize;
        }
        
        // 11. Test named_children() method
        auto namedChildren = moduleList->named_children();
        for (const auto& item : namedChildren) {
            auto name = item.key();
            auto child = item.value();
            auto paramSize = child->parameters().size();
            (void)name;
            (void)paramSize;
        }
        
        // 12. Test parameters() method
        auto params = moduleList->parameters();
        for (const auto& param : params) {
            auto paramShape = param.sizes();
            (void)paramShape;
        }
        
        // 13. Test named_parameters() method
        auto namedParams = moduleList->named_parameters();
        for (const auto& item : namedParams) {
            auto name = item.key();
            auto param = item.value();
            (void)name;
            (void)param;
        }
        
        // 14. Test clone() method
        if (moduleList->size() > 0) {
            auto clonedList = std::dynamic_pointer_cast<torch::nn::ModuleListImpl>(moduleList->clone());
            if (clonedList) {
                auto clonedSize = clonedList->size();
                (void)clonedSize;
            }
        }
        
        // 15. Test train() and eval() mode switching
        moduleList->train();
        moduleList->eval();
        
        // 16. Test to() method for device/dtype conversion
        moduleList->to(torch::kFloat32);
        
        // 17. Test zero_grad() method
        moduleList->zero_grad();
        
        // 18. Test at() method for indexed access (bounds-checked)
        if (moduleList->size() > 0 && offset < Size) {
            size_t idx = Data[offset] % moduleList->size();
            offset++;
            auto moduleAtIdx = moduleList->at<torch::nn::Module>(idx);
            (void)moduleAtIdx;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}