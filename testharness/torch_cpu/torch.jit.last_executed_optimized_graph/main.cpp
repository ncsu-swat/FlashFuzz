#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module with a forward method
        torch::jit::Module module("TestModule");
        
        // Define a simple script that uses the tensor
        std::string script_code = R"(
            def forward(self, x):
                return x + 1
        )";
        
        // Compile the module
        module.define(script_code);
        
        // Create inputs for the module
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Execute the module
        torch::jit::IValue output = module.forward(inputs);
        
        // Try to get the last executed optimized graph
        auto graph = torch::jit::lastExecutedOptimizedGraph();
        
        // If we got a graph, try to do something with it to ensure it's valid
        if (graph) {
            auto nodes = graph->nodes();
            for (auto it = nodes.begin(); it != nodes.end(); ++it) {
                auto node = *it;
                auto kind = node->kind();
                auto inputs = node->inputs();
                auto outputs = node->outputs();
            }
        }
        
        // Try running the module again with different inputs
        if (offset + 1 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            std::vector<torch::jit::IValue> more_inputs;
            more_inputs.push_back(another_tensor);
            
            // Execute the module again
            torch::jit::IValue another_output = module.forward(more_inputs);
            
            // Get the graph again
            auto another_graph = torch::jit::lastExecutedOptimizedGraph();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
