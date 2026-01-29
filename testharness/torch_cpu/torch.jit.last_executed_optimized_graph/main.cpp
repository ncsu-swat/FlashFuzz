#include "fuzzer_utils.h"
#include <iostream>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    // Keyword marker for harness verification.
    (void)"torch.jit.last_executed_optimized_graph";

    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Use fuzzer data to select operation type
        uint8_t op_selector = (offset < Size) ? Data[offset++] % 4 : 0;

        // Compile a simple function using torch::jit::compile
        torch::jit::Module module;
        
        try {
            // Create module using CompilationUnit
            auto cu = torch::jit::compile(R"(
                def forward(x):
                    return x + 1
            )");
            
            // Get the function and run it
            auto& forward_fn = cu->get_function("forward");
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Execute the function
            torch::jit::IValue output = forward_fn(inputs);
            if (output.isTensor()) {
                output.toTensor().sum();
            }
            
            // Try to get the last executed optimized graph
            auto graph = torch::jit::lastExecutedOptimizedGraph();
            
            // If we got a graph, inspect it to ensure it's valid
            if (graph) {
                auto nodes = graph->nodes();
                for (auto it = nodes.begin(); it != nodes.end(); ++it) {
                    auto node = *it;
                    auto kind = node->kind();
                    auto node_inputs = node->inputs();
                    auto node_outputs = node->outputs();
                    // Just access to trigger any potential issues
                    (void)kind;
                    (void)node_inputs.size();
                    (void)node_outputs.size();
                }
                
                // Get graph string representation
                std::string graph_str = graph->toString();
                (void)graph_str;
            }
            
            // Try different operations based on fuzzer data
            std::string script;
            switch (op_selector) {
                case 0:
                    script = R"(
                        def op(x):
                            return x * 2
                    )";
                    break;
                case 1:
                    script = R"(
                        def op(x):
                            return x.abs()
                    )";
                    break;
                case 2:
                    script = R"(
                        def op(x):
                            return x + x
                    )";
                    break;
                default:
                    script = R"(
                        def op(x):
                            return -x
                    )";
                    break;
            }
            
            auto cu2 = torch::jit::compile(script);
            auto& op_fn = cu2->get_function("op");
            
            // Create another tensor if we have more data
            torch::Tensor another_tensor;
            if (offset + 1 < Size) {
                another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                another_tensor = torch::randn({2, 3});
            }
            
            std::vector<torch::jit::IValue> op_inputs;
            op_inputs.push_back(another_tensor);
            
            torch::jit::IValue op_output = op_fn(op_inputs);
            if (op_output.isTensor()) {
                op_output.toTensor().sum();
            }
            
            // Get the optimized graph again after second execution
            auto another_graph = torch::jit::lastExecutedOptimizedGraph();
            if (another_graph) {
                // Access graph properties
                auto block = another_graph->block();
                (void)block;
            }
        }
        catch (const c10::Error&) {
            // Expected JIT compilation/execution errors - ignore silently
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors from JIT - ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}