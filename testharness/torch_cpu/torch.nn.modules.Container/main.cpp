#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a container module
        torch::nn::ModuleList container;
        
        // Determine how many tensors to create (1-4)
        uint8_t num_tensors = (Size > 0) ? (Data[0] % 4) + 1 : 1;
        offset++;
        
        // Create tensors and add them to the container
        std::vector<torch::Tensor> tensors;
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
                
                // Create a simple module to add to the container
                struct SimpleModule : torch::nn::Module {
                    SimpleModule() {}
                    torch::Tensor forward(torch::Tensor x) {
                        return x;
                    }
                };
                
                auto module = std::make_shared<SimpleModule>();
                container->push_back(module);
            } catch (const std::exception &e) {
                // Continue with the next tensor if one fails
                continue;
            }
        }
        
        // Test container operations
        if (container->size() > 0) {
            // Test size
            size_t container_size = container->size();
            
            // Test iteration
            for (const auto &module : *container) {
                if (!tensors.empty()) {
                    // Apply the module to a tensor
                    auto simple_module = std::dynamic_pointer_cast<torch::nn::Module>(module);
                    if (simple_module) {
                        // Cannot call forward on base Module, skip this test
                    }
                }
            }
            
            // Test indexing
            if (container_size > 0) {
                size_t idx = (Size > offset && offset < Size) ? (Data[offset] % container_size) : 0;
                auto module = (*container)[idx];
                
                if (!tensors.empty()) {
                    // Cannot call forward on base Module, skip this test
                }
            }
            
            // Test named modules
            auto named_modules = container->named_modules();
            for (const auto &pair : named_modules) {
                // Just access the name and module to test the functionality
                std::string name = pair.key();
                auto module = pair.value();
            }
            
            // Test named children
            auto named_children = container->named_children();
            for (const auto &pair : named_children) {
                std::string name = pair.key();
                auto child = pair.value();
            }
            
            // Test parameters
            auto parameters = container->parameters();
            
            // Test named parameters
            auto named_parameters = container->named_parameters();
            
            // Test buffers
            auto buffers = container->buffers();
            
            // Test named buffers
            auto named_buffers = container->named_buffers();
            
            // Test to() method for device conversion
            container->to(torch::kCPU);
            
            // Test train/eval modes
            container->train();
            container->eval();
        }
        
        // Test creating a sequential container
        torch::nn::Sequential sequential;
        
        // Add some modules to the sequential container
        for (uint8_t i = 0; i < std::min(num_tensors, (uint8_t)2) && i < tensors.size(); ++i) {
            struct SimpleModule : torch::nn::Module {
                SimpleModule() {}
                torch::Tensor forward(torch::Tensor x) {
                    return x;
                }
            };
            
            sequential->push_back(std::make_shared<SimpleModule>());
        }
        
        // Test sequential container forward
        if (sequential->size() > 0 && !tensors.empty()) {
            try {
                torch::Tensor result = sequential->forward(tensors[0]);
            } catch (const std::exception &) {
                // Ignore exceptions from forward pass
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
