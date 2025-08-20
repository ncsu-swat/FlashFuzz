#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if we have enough data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor using the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch.jit.is_scripting() function
        bool is_scripting = torch::jit::is_scripting();
        
        // Create a simple script module to test with
        torch::jit::Module module("test_module");
        
        // Create a scripted function that uses is_scripting
        std::string script_code = R"(
            def test_is_scripting(x):
                if torch.jit.is_scripting():
                    return x + 1
                else:
                    return x - 1
        )";
        
        // Try to compile the script
        module.define(script_code);
        
        // Create inputs for the scripted function
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        // Execute the scripted function
        torch::jit::IValue output;
        try {
            output = module.get_method("test_is_scripting")(inputs);
        } catch (...) {
            // Silently catch exceptions from executing the script
        }
        
        // Test direct call to is_scripting in different contexts
        bool is_scripting_again = torch::jit::is_scripting();
        
        // Try to create a TorchScript graph that uses is_scripting
        try {
            auto graph = std::make_shared<torch::jit::Graph>();
            auto result = graph->insertNode(graph->create(torch::jit::aten::is_scripting, {}));
            graph->registerOutput(result->output());
            
            auto schema = result->schema();
        } catch (...) {
            // Silently catch exceptions from graph creation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}