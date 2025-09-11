#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/csrc/jit/frontend/tracer.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple script module that uses the tensor
        torch::jit::script::Module module;
        
        try {
            // Create a traced function that will potentially trigger TracerWarning
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Define a function that might trigger tracer warnings
            auto traced_func = torch::jit::tracer::trace([&](torch::Tensor x) {
                // Operations that might trigger tracer warnings:
                
                // 1. Control flow that depends on tensor values
                if (x.dim() > 0 && x.size(0) > 0) {
                    auto first_val = x.flatten()[0];
                    if (first_val.item<float>() > 0) {
                        return x.sin();
                    } else {
                        return x.cos();
                    }
                }
                
                // 2. In-place operations
                if (x.numel() > 0) {
                    auto y = x.clone();
                    y.add_(1.0);
                    return y;
                }
                
                // 3. Random operations
                if (x.numel() == 0) {
                    return torch::rand_like(x);
                }
                
                // 4. Non-deterministic operations
                if (x.dim() > 1) {
                    return x.nonzero();
                }
                
                // 5. Size-dependent operations
                if (x.numel() > 10) {
                    return x.narrow(0, 0, x.size(0) / 2);
                }
                
                return x;
            }, inputs);
            
            // Try to use the traced function
            auto output = traced_func.forward(inputs);
            
            // Try to save the traced module
            traced_func.save("temp_module.pt");
            
            // Try to load the traced module
            module = torch::jit::load("temp_module.pt");
            
            // Execute the loaded module
            auto result = module.forward(inputs);
        }
        catch (const c10::Warning& w) {
            // This is expected in some cases, so we'll just continue
        }
        
        // Try another approach that might trigger TracerWarning
        try {
            // Create a module with control flow
            torch::jit::script::Module script_module;
            
            // Register a method with control flow
            auto script_method = torch::jit::script::CompilationUnit().define(
                "forward", R"(
                    def forward(self, x):
                        if x.size(0) > 0:
                            return x.sin()
                        else:
                            return x.cos()
                )", c10::nullopt, nullptr);
            
            // Try to trace this method
            std::vector<torch::jit::IValue> script_inputs;
            script_inputs.push_back(input_tensor);
            
            // This might trigger TracerWarning
            auto traced_script = torch::jit::tracer::trace(script_method, script_inputs);
        }
        catch (const c10::Warning& w) {
            // Expected in some cases
        }
        
        // Try to trigger TracerWarning with dynamic shapes
        try {
            auto dynamic_func = [](torch::Tensor x) {
                auto shape = x.sizes().vec();
                if (!shape.empty()) {
                    shape[0] = shape[0] * 2;
                    return x.reshape(shape);
                }
                return x;
            };
            
            std::vector<torch::jit::IValue> dynamic_inputs;
            dynamic_inputs.push_back(input_tensor);
            
            auto traced_dynamic = torch::jit::tracer::trace(dynamic_func, dynamic_inputs);
            
            // Try with a different shape
            if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                auto new_tensor = input_tensor.narrow(0, 0, input_tensor.size(0) - 1);
                std::vector<torch::jit::IValue> new_inputs;
                new_inputs.push_back(new_tensor);
                
                auto output = traced_dynamic.forward(new_inputs);
            }
        }
        catch (const c10::Warning& w) {
            // Expected in some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
