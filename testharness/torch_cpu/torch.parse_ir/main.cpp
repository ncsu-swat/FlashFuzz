#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/csrc/jit/ir/irparser.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a string from the input data to use as IR
        size_t string_length = std::min(Size, static_cast<size_t>(4096)); // Limit string size
        std::string ir_string(reinterpret_cast<const char*>(Data), string_length);
        
        // Try to parse the IR string into a new graph
        try {
            auto graph = std::make_shared<torch::jit::Graph>();
            torch::jit::parseIR(ir_string, graph.get());
        } catch (...) {
            // Parsing errors are expected for random/malformed data
        }
        
        // Try parsing with different string prefixes that might be valid IR
        if (Size > 50) {
            try {
                // TorchScript IR typically starts with "graph"
                std::string prefixed_ir = "graph():\n  " + ir_string;
                auto graph = std::make_shared<torch::jit::Graph>();
                torch::jit::parseIR(prefixed_ir, graph.get());
            } catch (...) {
                // Parsing errors are expected
            }
        }
        
        // Try with a more structured IR template
        if (Size > 20) {
            try {
                // Create a simple graph structure with the fuzzed content as an operation
                std::string structured_ir = "graph(%x : Tensor):\n";
                
                // Use first byte to select an operation type
                uint8_t op_selector = Data[0] % 5;
                switch (op_selector) {
                    case 0:
                        structured_ir += "  %r = aten::relu(%x)\n";
                        break;
                    case 1:
                        structured_ir += "  %r = aten::sigmoid(%x)\n";
                        break;
                    case 2:
                        structured_ir += "  %r = aten::tanh(%x)\n";
                        break;
                    case 3:
                        structured_ir += "  %r = aten::neg(%x)\n";
                        break;
                    default:
                        structured_ir += "  %r = aten::abs(%x)\n";
                        break;
                }
                structured_ir += "  return (%r)\n";
                
                auto graph = std::make_shared<torch::jit::Graph>();
                torch::jit::parseIR(structured_ir, graph.get());
            } catch (...) {
                // Parsing errors are expected
            }
        }
        
        // Try parsing with fuzzed graph structure
        if (Size > 100) {
            try {
                // Extract parts of fuzzed data for graph components
                size_t name_len = std::min(static_cast<size_t>(Data[0] % 32 + 1), Size - 1);
                std::string var_name(reinterpret_cast<const char*>(Data + 1), name_len);
                
                // Sanitize variable name (only alphanumeric and underscore)
                for (char& c : var_name) {
                    if (!std::isalnum(c) && c != '_') {
                        c = '_';
                    }
                }
                if (!var_name.empty() && std::isdigit(var_name[0])) {
                    var_name[0] = 'v';
                }
                if (var_name.empty()) {
                    var_name = "x";
                }
                
                std::string fuzzed_graph = "graph(%" + var_name + " : Tensor):\n";
                fuzzed_graph += "  %out = aten::clone(%" + var_name + ")\n";
                fuzzed_graph += "  return (%out)\n";
                
                auto graph = std::make_shared<torch::jit::Graph>();
                torch::jit::parseIR(fuzzed_graph, graph.get());
            } catch (...) {
                // Parsing errors are expected
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}