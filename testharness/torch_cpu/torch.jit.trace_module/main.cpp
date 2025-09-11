#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

struct TestModule : torch::nn::Module {
    TestModule() {
        register_module("linear", linear);
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear(x);
    }

    torch::nn::Linear linear{10, 5};
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module to trace
        TestModule module;
        
        // Create a dictionary of inputs for tracing
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Create a map of methods to trace
        std::unordered_map<std::string, std::vector<torch::jit::IValue>> method_inputs;
        method_inputs["forward"] = inputs;
        
        // Try to trace the module
        torch::jit::script::Module traced_module;
        try {
            traced_module = torch::jit::trace_module(std::make_shared<TestModule>(module), method_inputs);
            
            // Test the traced module with the same input
            auto output = traced_module.forward(inputs);
            
            // Try with a different input if there's enough data
            if (offset + 4 < Size) {
                torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
                std::vector<torch::jit::IValue> another_inputs;
                another_inputs.push_back(another_input);
                auto another_output = traced_module.forward(another_inputs);
            }
            
            // Try to save and load the traced module
            if (Size % 3 == 0) {
                traced_module.save("temp_module.pt");
                auto loaded_module = torch::jit::load("temp_module.pt");
                auto loaded_output = loaded_module.forward(inputs);
            }
        } catch (const c10::Error& e) {
            // This is a normal PyTorch error, not a bug in our code
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
