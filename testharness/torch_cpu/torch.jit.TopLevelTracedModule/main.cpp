#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module to trace
        struct SimpleModule : torch::nn::Module {
            torch::Tensor forward(torch::Tensor x) {
                return x.sigmoid();
            }
        };
        
        // Create an instance of the module
        SimpleModule module;
        
        // Create a traced module
        try {
            // Create inputs for tracing
            std::vector<torch::jit::IValue> trace_inputs;
            trace_inputs.push_back(input_tensor);
            
            // Trace the module using torch::jit::trace_module
            auto traced_module = torch::jit::trace_module(module, {{"forward", trace_inputs}});
            
            // Test the traced module with the same input
            std::vector<torch::jit::IValue> test_inputs;
            test_inputs.push_back(input_tensor);
            
            // Run the traced module
            torch::Tensor output = traced_module.forward(test_inputs).toTensor();
            
            // Try with different input shapes if we have enough data
            if (Size - offset > 4) {
                torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try with different input if possible
                try {
                    std::vector<torch::jit::IValue> another_inputs;
                    another_inputs.push_back(another_input);
                    torch::Tensor another_output = traced_module.forward(another_inputs).toTensor();
                } catch (...) {
                    // Ignore errors with different shaped inputs
                }
            }
            
            // Try serializing and deserializing the module
            std::stringstream ss;
            traced_module.save(ss);
            
            // Load the module back
            torch::jit::script::Module loaded_module = torch::jit::load(ss);
            
            // Run the loaded module
            torch::Tensor loaded_output = loaded_module.forward(test_inputs).toTensor();
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}