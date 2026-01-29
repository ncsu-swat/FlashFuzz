#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

// Define TestModule outside LLVMFuzzerTestOneInput for cleaner code
struct TestModule : torch::nn::Module {
    TestModule() {
        param = register_parameter("param", torch::randn({3, 3}));
        register_buffer("buffer", torch::ones({2, 2}));
        submodule = register_module("submodule", torch::nn::Linear(10, 5));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        if (x.dim() > 0 && x.size(0) > 0) {
            return x + param.sum();
        }
        return x;
    }
    
    torch::Tensor param;
    torch::nn::Linear submodule{nullptr};
};

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
        
        // Create a tensor to use with the module
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create an instance of the module
        auto module = std::make_shared<TestModule>();
        
        // Test various module operations
        if (offset < Size) {
            uint8_t op_selector = Data[offset++] % 14;
            
            switch (op_selector) {
                case 0: {
                    // Test forward pass
                    torch::Tensor output = module->forward(input_tensor);
                    break;
                }
                case 1: {
                    // Test parameter access
                    auto params = module->parameters();
                    if (!params.empty()) {
                        torch::Tensor param = params[0];
                    }
                    break;
                }
                case 2: {
                    // Test buffer access
                    auto buffers = module->buffers();
                    if (!buffers.empty()) {
                        torch::Tensor buffer = buffers[0];
                    }
                    break;
                }
                case 3: {
                    // Test named parameter access
                    auto named_params = module->named_parameters();
                    for (auto& param : named_params) {
                        std::string name = param.key();
                        torch::Tensor tensor = param.value();
                    }
                    break;
                }
                case 4: {
                    // Test named buffer access
                    auto named_buffers = module->named_buffers();
                    for (auto& buffer : named_buffers) {
                        std::string name = buffer.key();
                        torch::Tensor tensor = buffer.value();
                    }
                    break;
                }
                case 5: {
                    // Test children modules
                    auto children = module->children();
                    for (auto& child : children) {
                        auto child_params = child->parameters();
                    }
                    break;
                }
                case 6: {
                    // Test named children
                    auto named_children = module->named_children();
                    for (auto& child : named_children) {
                        std::string name = child.key();
                        auto child_module = child.value();
                    }
                    break;
                }
                case 7: {
                    // Test module training mode
                    bool was_training = module->is_training();
                    module->train(!was_training);
                    module->train(was_training);
                    break;
                }
                case 8: {
                    // Test module evaluation mode
                    module->eval();
                    break;
                }
                case 9: {
                    // Test module to device
                    module->to(torch::kCPU);
                    break;
                }
                case 10: {
                    // Test modules() recursive traversal
                    auto all_modules = module->modules();
                    for (auto& mod : all_modules) {
                        mod->is_training();
                    }
                    break;
                }
                case 11: {
                    // Test named_modules() recursive traversal
                    auto named_modules = module->named_modules();
                    for (auto& item : named_modules) {
                        std::string name = item.key();
                        auto mod = item.value();
                    }
                    break;
                }
                case 12: {
                    // Test apply() for recursive function application
                    module->apply([](torch::nn::Module& m) {
                        m.train(true);
                    });
                    break;
                }
                case 13: {
                    // Test to() with dtype
                    module->to(torch::kFloat32);
                    break;
                }
            }
        }
        
        // Test additional module functionality if we have more data
        if (offset + 1 < Size) {
            uint8_t extra_op = Data[offset++] % 5;
            
            switch (extra_op) {
                case 0: {
                    // Test module zero_grad
                    module->zero_grad();
                    break;
                }
                case 1: {
                    // Test module name
                    std::string name = module->name();
                    break;
                }
                case 2: {
                    // Test module parameters update
                    for (auto& param : module->parameters()) {
                        if (param.requires_grad()) {
                            param.add_(torch::randn_like(param) * 0.01);
                        }
                    }
                    break;
                }
                case 3: {
                    // Test parameters with recurse=false
                    auto params = module->parameters(/*recurse=*/false);
                    break;
                }
                case 4: {
                    // Test buffers with recurse=false
                    auto buffers = module->buffers(/*recurse=*/false);
                    break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}