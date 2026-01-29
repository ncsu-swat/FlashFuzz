#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract batch size from fuzzer data (1-16)
        uint8_t batch_size = (Data[offset++] % 16) + 1;
        
        // Extract input features from fuzzer data
        uint8_t in_features = (Data[offset++] % 32) + 1;
        uint8_t out_features = (Data[offset++] % 32) + 1;

        // Create a simple traced module using TorchScript
        // torch::jit::trace works on functions/modules
        // For fuzzing trace_module, we need to work with existing traced modules
        
        // Create input tensor with appropriate shape
        torch::Tensor input_tensor = torch::randn({batch_size, in_features});
        
        // Use remaining fuzzer data to perturb the tensor values
        size_t remaining = Size - offset;
        if (remaining > 0) {
            auto accessor = input_tensor.accessor<float, 2>();
            size_t data_idx = 0;
            for (int64_t i = 0; i < batch_size && data_idx < remaining; i++) {
                for (int64_t j = 0; j < in_features && data_idx < remaining; j++) {
                    // Use fuzzer bytes to create float values
                    float val = static_cast<float>(Data[offset + data_idx]) / 255.0f * 2.0f - 1.0f;
                    accessor[i][j] = val;
                    data_idx++;
                }
            }
        }

        // Create a simple module using TorchScript compilation
        // This is the proper way to create traceable modules in C++
        std::string module_src = R"(
            def forward(self, x):
                return x * 2 + 1
        )";
        
        // Create a script module that we can trace
        torch::jit::Module script_module("TestModule");
        script_module.define(module_src);
        
        // Prepare inputs for tracing
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Create method inputs map for trace_module
        c10::Dict<std::string, std::vector<torch::jit::IValue>> method_inputs;
        method_inputs.insert("forward", inputs);
        
        try {
            // trace_module traces an existing Module's methods
            auto traced = torch::jit::trace_module(script_module, method_inputs);
            
            // Run the traced module
            auto output = traced.forward(inputs);
            
            // Verify output is valid
            if (output.isTensor()) {
                torch::Tensor out_tensor = output.toTensor();
                // Force computation
                (void)out_tensor.sum().item<float>();
            }
            
            // Try with different input shape based on fuzzer data
            if (Size > 10 && Data[Size - 1] % 2 == 0) {
                uint8_t new_batch = (Data[Size - 2] % 8) + 1;
                torch::Tensor new_input = torch::randn({new_batch, in_features});
                std::vector<torch::jit::IValue> new_inputs;
                new_inputs.push_back(new_input);
                auto new_output = traced.forward(new_inputs);
            }
            
            // Test cloning the traced module
            if (Size > 5 && Data[4] % 3 == 0) {
                auto cloned = traced.clone();
                auto clone_output = cloned.forward(inputs);
            }
            
        } catch (const c10::Error& e) {
            // Expected PyTorch errors (shape mismatch, etc.) - not bugs
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors from tracing are expected for some inputs
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}