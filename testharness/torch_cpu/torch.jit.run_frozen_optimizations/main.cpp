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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a simple module with a function
        torch::jit::Module module("test_module");
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple script function that returns the input tensor
        std::string script_code = "def forward(self, x):\n    return x";
        
        // Define the module with the script
        module.define(script_code);
        
        // Create a method schema for the forward method
        std::vector<c10::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Get the method
        auto method = module.get_method("forward");
        
        // Create a graph from the method
        std::shared_ptr<torch::jit::Graph> graph = method.graph()->copy();
        
        // Run frozen optimizations on the graph (using FrozenModule approach)
        torch::jit::Module frozen_module = torch::jit::freeze(module);
        
        // Try to execute the frozen module
        auto result = frozen_module.forward(inputs);
        
        // Try to access the result as a tensor
        if (result.isTensor()) {
            torch::Tensor output = result.toTensor();
            
            // Perform some operations on the output to ensure it's valid
            if (output.defined()) {
                auto sum = output.sum();
                auto mean = output.mean();
                auto max_val = output.max();
            }
        }
        
        // Try with different optimization approaches
        if (offset < Size) {
            uint8_t opt_level = Data[offset++] % 3;  // 0, 1, or 2
            
            // Create a new graph copy
            std::shared_ptr<torch::jit::Graph> graph2 = method.graph()->copy();
            
            // Run basic graph optimizations
            torch::jit::EliminateDeadCode(graph2);
            torch::jit::EliminateCommonSubexpression(graph2);
            
            // Create executor and run
            torch::jit::GraphExecutor executor2(graph2, "");
            auto stack2 = torch::jit::Stack();
            for (const auto& input : inputs) {
                torch::jit::push(stack2, input);
            }
            
            executor2.run(stack2);
        }
        
        // Try with a more complex model if we have enough data
        if (Size - offset > 10) {
            // Create a more complex module
            torch::jit::Module complex_module("complex_module");
            
            // Create a more complex script with operations
            std::string complex_script = 
                "def forward(self, x):\n"
                "    y = x * 2.0\n"
                "    z = torch.relu(y)\n"
                "    return z + x\n";
            
            complex_module.define(complex_script);
            
            // Freeze the complex module for optimization
            torch::jit::Module frozen_complex = torch::jit::freeze(complex_module);
            
            // Execute the frozen complex module
            auto complex_result = frozen_complex.forward(inputs);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
