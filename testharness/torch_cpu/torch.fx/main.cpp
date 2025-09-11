#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/passes/optimize_for_inference.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>

// Simple module for testing FX functionality
struct TestModule : torch::nn::Module {
    TestModule() {
        register_parameter("weight", torch::randn({10, 10}));
        register_parameter("bias", torch::randn({10}));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::matmul(x, weight) + bias;
    }

    torch::Tensor weight, bias;
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a test module
        TestModule module;
        
        // Try to trace the module with the input tensor
        try {
            auto traced_module = torch::jit::trace(module, {input});
            
            // Test running the traced module
            auto output = traced_module.forward({input});
            
            // Test serialization/deserialization
            if (offset + 1 < Size) {
                uint8_t serialize_flag = Data[offset++];
                if (serialize_flag % 2 == 0) {
                    std::stringstream ss;
                    traced_module.save(ss);
                    auto loaded_module = torch::jit::load(ss);
                    auto loaded_output = loaded_module.forward({input});
                }
            }
            
            // Test graph manipulation if we have more data
            if (offset + 1 < Size) {
                uint8_t graph_flag = Data[offset++];
                if (graph_flag % 3 == 0) {
                    auto graph = traced_module.get_method("forward").graph();
                    for (auto node : graph->nodes()) {
                        auto kind = node->kind();
                    }
                }
            }
            
            // Test optimization if we have more data
            if (offset + 1 < Size) {
                uint8_t opt_flag = Data[offset++];
                if (opt_flag % 2 == 0) {
                    traced_module = torch::jit::optimize_for_inference(traced_module);
                    auto optimized_output = traced_module.forward({input});
                }
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't discard the input
            return 0;
        }
        
        // Try to create a JIT script module
        try {
            std::string script_code = R"(
                def forward(self, x):
                    return torch.matmul(x, self.weight) + self.bias
            )";
            
            auto script_module = torch::jit::compile(script_code);
            
            // Test graph manipulation
            if (offset + 1 < Size) {
                uint8_t graph_flag = Data[offset++];
                if (graph_flag % 2 == 0) {
                    auto graph = script_module.get_method("forward").graph();
                    for (auto node : graph->nodes()) {
                        auto kind = node->kind();
                    }
                }
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't discard the input
            return 0;
        }
        
        // Create a second tensor if we have more data
        if (offset + 2 < Size) {
            try {
                torch::Tensor second_input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test with multiple inputs - create a simple module that accepts two inputs
                auto simple_module = torch::nn::Sequential(
                    torch::nn::Linear(10, 5),
                    torch::nn::ReLU()
                );
                
                auto traced_multi = torch::jit::trace(simple_module, input);
                
                // Test different execution modes
                if (offset + 1 < Size) {
                    uint8_t exec_flag = Data[offset++];
                    if (exec_flag % 3 == 0) {
                        traced_multi.eval();
                    } else if (exec_flag % 3 == 1) {
                        traced_multi.train();
                    }
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors but don't discard the input
                return 0;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
