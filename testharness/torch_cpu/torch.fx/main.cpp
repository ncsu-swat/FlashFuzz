#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For serialization streams
#include <algorithm>      // For std::min
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

// Simple module for testing FX functionality
// Target API: torch.fx
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
        auto float_input = input.to(torch::kFloat);
        auto flat = float_input.flatten();
        int64_t feature_size = flat.numel();
        if (feature_size <= 0) {
            feature_size = 1;
            flat = torch::zeros({feature_size}, float_input.options());
        }
        feature_size = std::min<int64_t>(feature_size, 16);
        auto trimmed = flat.narrow(0, 0, feature_size).contiguous();
        auto normalized = trimmed.view({1, feature_size});
        
        // Create a small script via the JIT API and run it
        std::string script_source = R"JIT(
            def forward(x, w, b):
                y = torch.matmul(x, w) + b
                return torch.relu(y)
        )JIT";
        auto cu = torch::jit::compile(script_source);
        auto weight =
            torch::randn({feature_size, feature_size}, normalized.options());
        auto bias = torch::randn({feature_size}, normalized.options());
        auto output_ivalue =
            cu->run_method("forward", normalized, weight, bias);
        auto output = output_ivalue.toTensor();
        output.sum().item<float>();
        
        // Inspect and run the graph through GraphExecutor when available
        if (offset + 1 < Size) {
            uint8_t graph_flag = Data[offset++];
            auto fn = cu->find_function(c10::QualifiedName("forward"));
            auto* graph_fn = fn ? torch::jit::tryToGraphFunction(*fn) : nullptr;
            if (graph_fn && graph_flag % 2 == 0) {
                auto graph_copy = graph_fn->graph()->copy();
                torch::jit::GraphExecutor executor(graph_copy, "fx_graph_executor");
                torch::jit::Stack stack{normalized, weight, bias};
                executor.run(stack);
                if (!stack.empty() && stack.back().isTensor()) {
                    stack.back().toTensor().sum().item<float>();
                }
            } else if (graph_fn) {
                for (auto node : graph_fn->graph()->nodes()) {
                    (void)node->kind();
                }
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
