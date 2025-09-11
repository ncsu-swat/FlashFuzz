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
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module
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
                // Simple operation that uses the parameter
                if (x.dim() > 0 && x.size(0) > 0) {
                    x = torch::nn::functional::relu(x);
                    
                    // Reshape if needed to match linear layer input
                    if (x.dim() > 1) {
                        auto batch_size = x.size(0);
                        x = x.reshape({batch_size, -1});
                    } else {
                        x = x.reshape({1, -1});
                    }
                    
                    // Ensure the last dimension is 10 for the linear layer
                    if (x.size(-1) != 10) {
                        x = x.expand({x.size(0), 10});
                    }
                    
                    x = submodule->forward(x);
                }
                return x;
            }
            
            torch::Tensor param;
            torch::nn::Linear submodule{nullptr};
        };
        
        // Create an instance of the module
        auto module = std::make_shared<SimpleModule>();
        
        // Test module state
        module->train();
        module->eval();
        
        // Test parameter access
        auto params = module->parameters();
        auto buffers = module->buffers();
        auto named_params = module->named_parameters();
        auto named_buffers = module->named_buffers();
        
        // Test module methods
        module->to(torch::kCPU);
        module->to(torch::kFloat);
        
        // Apply the module to the input tensor
        torch::Tensor output = module->forward(input);
        
        // Test module serialization if we have enough data
        if (Size > offset + 10) {
            torch::save(module, "temp_module.pt");
            torch::load(module, "temp_module.pt");
        }
        
        // Test module children and named children
        auto children = module->children();
        auto named_children = module->named_children();
        
        // Test module zero_grad
        module->zero_grad();
        
        // Test module apply
        module->apply([](torch::nn::Module& m) {
            if (auto linear = dynamic_cast<torch::nn::LinearImpl*>(&m)) {
                linear->reset_parameters();
            }
            return m;
        });
        
        // Test module with different input types if we have more data
        if (Size > offset + 20) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            output = module->forward(input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
