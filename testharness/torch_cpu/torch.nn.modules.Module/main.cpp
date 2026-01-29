#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module to test torch::nn::Module functionality
        struct SimpleModule : torch::nn::Module {
            SimpleModule() {
                // Register a parameter
                param = register_parameter("param", torch::randn({4, 4}));
                
                // Register a buffer
                register_buffer("buffer", torch::ones({2, 2}));
                
                // Register a submodule
                submodule = register_module("submodule", torch::nn::Linear(10, 5));
            }
            
            torch::Tensor forward(torch::Tensor x) {
                // Apply relu activation
                x = torch::relu(x);
                
                // Flatten and pad/truncate to match linear layer input size
                x = x.flatten();
                int64_t numel = x.numel();
                
                if (numel == 0) {
                    return torch::zeros({1, 5});
                }
                
                // Pad or truncate to size 10
                if (numel < 10) {
                    x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 10 - numel}));
                } else if (numel > 10) {
                    x = x.slice(0, 0, 10);
                }
                
                x = x.reshape({1, 10});
                x = submodule->forward(x);
                
                return x;
            }
            
            torch::Tensor param;
            torch::nn::Linear submodule{nullptr};
        };
        
        // Create an instance of the module
        auto module = std::make_shared<SimpleModule>();
        
        // Test train/eval mode switching
        module->train();
        bool is_training = module->is_training();
        (void)is_training;
        
        module->eval();
        is_training = module->is_training();
        (void)is_training;
        
        // Test parameter access
        auto params = module->parameters();
        auto buffers = module->buffers();
        auto named_params = module->named_parameters();
        auto named_buffers = module->named_buffers();
        
        // Iterate over parameters
        for (auto& p : params) {
            (void)p.numel();
        }
        
        // Test named parameters
        for (auto& np : named_params) {
            (void)np.key();
            (void)np.value().numel();
        }
        
        // Test module methods
        module->to(torch::kCPU);
        module->to(torch::kFloat32);
        module->to(torch::kFloat64);
        module->to(torch::kFloat32);  // Convert back
        
        // Apply the module to the input tensor (wrap in try-catch for shape issues)
        torch::Tensor output;
        try {
            output = module->forward(input.to(torch::kFloat32));
        } catch (...) {
            // Shape mismatch or other expected errors, ignore
        }
        
        // Test module children and named children
        auto children = module->children();
        auto named_children = module->named_children();
        
        for (auto& child : children) {
            (void)child->is_training();
        }
        
        for (auto& nc : named_children) {
            (void)nc.key();
        }
        
        // Test module zero_grad
        module->zero_grad();
        
        // Verify gradients are zeroed
        for (auto& p : module->parameters()) {
            if (p.grad().defined()) {
                (void)p.grad().sum().item<float>();
            }
        }
        
        // Test module apply (lambda takes Module& and returns void)
        module->apply([](torch::nn::Module& m) {
            (void)m.is_training();
        });
        
        // Test clone (if available)
        try {
            auto cloned = module->clone();
            (void)cloned;
        } catch (...) {
            // clone may not be implemented
        }
        
        // Test pretty_print
        std::ostringstream oss;
        module->pretty_print(oss);
        (void)oss.str();
        
        // Test with another input if we have more data
        if (Size > offset + 8) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                output = module->forward(input2.to(torch::kFloat32));
            } catch (...) {
                // Expected shape issues, ignore
            }
        }
        
        // Test requires_grad setting
        for (auto& p : module->parameters()) {
            p.set_requires_grad(false);
            p.set_requires_grad(true);
        }
        
        // Test named_modules
        auto named_modules = module->named_modules();
        for (auto& nm : named_modules) {
            (void)nm.key();
            (void)nm.value()->is_training();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}