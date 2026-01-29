#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

// Simple module for tracing
struct SimpleModule : torch::nn::Module {
    SimpleModule() {
        // Register a simple linear layer
        linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(10, 10)));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }
    
    torch::nn::Linear linear{nullptr};
};

// Another simple module that does element-wise operations
struct ElementWiseModule : torch::nn::Module {
    torch::Tensor forward(torch::Tensor x) {
        return x * 2 + 1;
    }
};

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
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Get module choice from fuzz data
        uint8_t module_choice = Data[offset++] % 2;
        
        // Create input tensor for tracing
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a valid tensor for tracing
        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            return 0;
        }
        
        // Convert to float for module operations
        input_tensor = input_tensor.to(torch::kFloat32);
        
        if (module_choice == 0) {
            // Test with ElementWiseModule (works with any shape)
            auto module = std::make_shared<ElementWiseModule>();
            module->eval();
            
            try {
                // Trace the module
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                torch::jit::script::Module traced_module = torch::jit::trace(*module, inputs);
                
                // Run the traced module with the same input
                torch::jit::IValue output = traced_module.forward(inputs);
                
                if (output.isTensor()) {
                    torch::Tensor output_tensor = output.toTensor();
                    (void)output_tensor.sizes();
                }
                
                // Test with a different input of the same shape
                if (offset < Size) {
                    torch::Tensor second_input = fuzzer_utils::createTensor(Data, Size, offset);
                    if (second_input.defined() && second_input.numel() > 0) {
                        second_input = second_input.to(torch::kFloat32);
                        // Reshape to match original input shape if possible
                        try {
                            second_input = second_input.reshape(input_tensor.sizes());
                            std::vector<torch::jit::IValue> second_inputs;
                            second_inputs.push_back(second_input);
                            torch::jit::IValue second_output = traced_module.forward(second_inputs);
                        } catch (...) {
                            // Shape mismatch is expected
                        }
                    }
                }
            } catch (const c10::Error&) {
                // Expected PyTorch errors are acceptable
            } catch (const std::runtime_error&) {
                // Expected runtime errors
            }
        } else {
            // Test with SimpleModule (requires specific input shape)
            auto module = std::make_shared<SimpleModule>();
            module->eval();
            
            try {
                // Reshape input to work with the linear layer (batch_size, 10)
                int64_t numel = input_tensor.numel();
                int64_t batch_size = std::max(int64_t(1), numel / 10);
                torch::Tensor reshaped_input = input_tensor.reshape({-1}).narrow(0, 0, std::min(numel, batch_size * 10));
                reshaped_input = reshaped_input.reshape({batch_size, 10});
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(reshaped_input);
                
                // Trace the module
                torch::jit::script::Module traced_module = torch::jit::trace(*module, inputs);
                
                // Run the traced module
                torch::jit::IValue output = traced_module.forward(inputs);
                
                if (output.isTensor()) {
                    torch::Tensor output_tensor = output.toTensor();
                    (void)output_tensor.sizes();
                }
            } catch (const c10::Error&) {
                // Expected PyTorch errors
            } catch (const std::runtime_error&) {
                // Expected runtime errors
            }
        }
        
        // Test tracing with check_trace option variations
        if (offset < Size) {
            uint8_t check_trace = Data[offset++] % 2;
            
            try {
                auto module = std::make_shared<ElementWiseModule>();
                module->eval();
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Trace with different settings
                torch::jit::script::Module traced = torch::jit::trace(*module, inputs);
                
                // Clone and save/load test
                if (check_trace) {
                    auto cloned = traced.clone();
                    torch::jit::IValue clone_output = cloned.forward(inputs);
                }
            } catch (const c10::Error&) {
                // Expected errors
            } catch (const std::runtime_error&) {
                // Expected errors
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