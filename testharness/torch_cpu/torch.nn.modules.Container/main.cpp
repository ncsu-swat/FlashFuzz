#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

// Define a simple module that can be registered with ModuleList/Sequential
struct IdentityModuleImpl : torch::nn::Module {
    IdentityModuleImpl() {}
    torch::Tensor forward(torch::Tensor x) {
        return x;
    }
};
TORCH_MODULE(IdentityModule);

struct ScaleModuleImpl : torch::nn::Module {
    double scale;
    ScaleModuleImpl(double s = 1.0) : scale(s) {}
    torch::Tensor forward(torch::Tensor x) {
        return x * scale;
    }
};
TORCH_MODULE(ScaleModule);

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
        
        // Create a ModuleList container
        torch::nn::ModuleList container;
        
        // Determine how many modules to create (1-4)
        uint8_t num_modules = (Data[offset] % 4) + 1;
        offset++;
        
        // Create tensors for testing
        std::vector<torch::Tensor> tensors;
        for (uint8_t i = 0; i < num_modules && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception &) {
                continue;
            }
        }
        
        // Add modules to the container
        for (uint8_t i = 0; i < num_modules; ++i) {
            if (offset < Size && Data[offset % Size] % 2 == 0) {
                container->push_back(IdentityModule());
            } else {
                double scale = (offset < Size) ? (double)(Data[offset % Size] % 10 + 1) / 5.0 : 1.0;
                container->push_back(ScaleModule(scale));
            }
            offset++;
        }
        
        // Test container operations
        if (container->size() > 0) {
            // Test size
            size_t container_size = container->size();
            (void)container_size;
            
            // Test iteration
            for (const auto &module : *container) {
                (void)module;
            }
            
            // Test indexing
            if (container->size() > 0 && offset < Size) {
                size_t idx = Data[offset] % container->size();
                offset++;
                auto module = container[idx];
                (void)module;
            }
            
            // Test named modules
            auto named_modules = container->named_modules();
            for (const auto &pair : named_modules) {
                std::string name = pair.key();
                (void)name;
            }
            
            // Test named children
            auto named_children = container->named_children();
            for (const auto &pair : named_children) {
                std::string name = pair.key();
                (void)name;
            }
            
            // Test parameters
            auto parameters = container->parameters();
            (void)parameters;
            
            // Test named parameters
            auto named_parameters = container->named_parameters();
            (void)named_parameters;
            
            // Test buffers
            auto buffers = container->buffers();
            (void)buffers;
            
            // Test named buffers
            auto named_buffers = container->named_buffers();
            (void)named_buffers;
            
            // Test to() method for device conversion
            container->to(torch::kCPU);
            container->to(torch::kFloat32);
            
            // Test train/eval modes
            container->train();
            bool is_training = container->is_training();
            (void)is_training;
            container->eval();
            is_training = container->is_training();
            (void)is_training;
            
            // Test clone
            auto cloned = container->clone();
            (void)cloned;
        }
        
        // Test ModuleDict container
        torch::nn::ModuleDict dict;
        
        // Use insert() method instead of update() with initializer list
        dict->insert("identity", IdentityModule());
        if (offset < Size && Size > 1) {
            double scale = (double)(Data[offset % Size] % 10 + 1) / 5.0;
            dict->insert("scale", ScaleModule(scale));
        }
        
        // Test ModuleDict operations
        if (dict->size() > 0) {
            // Test contains
            bool has_identity = dict->contains("identity");
            (void)has_identity;
            
            // Test keys
            auto keys = dict->keys();
            for (const auto &key : keys) {
                (void)key;
            }
            
            // Test iteration
            for (const auto &item : *dict) {
                std::string name = item.key();
                (void)name;
            }
        }
        
        // Test ParameterList
        torch::nn::ParameterList param_list;
        if (!tensors.empty()) {
            for (size_t i = 0; i < std::min(tensors.size(), (size_t)3); ++i) {
                try {
                    torch::Tensor t = tensors[i].clone().detach().requires_grad_(false);
                    param_list->append(t);
                } catch (const std::exception &) {
                    // Skip invalid tensors
                }
            }
        }
        
        if (param_list->size() > 0) {
            size_t plist_size = param_list->size();
            (void)plist_size;
        }
        
        // Test Sequential container
        torch::nn::Sequential sequential(
            IdentityModule(),
            ScaleModule(1.5)
        );
        
        // Test sequential operations
        sequential->push_back(IdentityModule());
        
        // Test sequential forward with a valid tensor
        if (!tensors.empty()) {
            try {
                // Create a simple float tensor for forward pass
                torch::Tensor input = torch::randn({2, 3});
                torch::Tensor result = sequential->forward(input);
                (void)result;
            } catch (const std::exception &) {
                // Shape mismatch or other issues - expected
            }
        }
        
        // Test train/eval on sequential
        sequential->train();
        sequential->eval();
        
        // Test zero_grad
        container->zero_grad();
        sequential->zero_grad();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}