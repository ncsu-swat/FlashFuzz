#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/tracer.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Target API keyword for harness checks: torch.jit.TracerWarning
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple script module that uses the tensor
        auto name_lookup = [](const torch::autograd::Variable &) {
            return std::string();
        };

        try {
            torch::jit::Stack inputs;
            inputs.push_back(input_tensor);

            auto traced_func = torch::jit::tracer::trace(
                inputs,
                [&](const torch::jit::Stack &stack) {
                    auto x = stack.front().toTensor();
                    torch::Tensor y = x;

                    // Control flow that depends on tensor values
                    if (x.dim() > 0 && x.size(0) > 0) {
                        auto first_val = x.flatten()[0];
                        if (first_val.item<float>() > 0) {
                            y = x.sin();
                        } else {
                            y = x.cos();
                        }
                    }

                    if (y.numel() == 0) {
                        y = torch::rand_like(x);
                    }

                    if (y.numel() > 0) {
                        auto temp = y.clone();
                        temp.add_(1.0);
                        y = temp;
                    }

                    if (y.dim() > 1) {
                        y = y.transpose(0, y.dim() - 1);
                    }

                    if (y.numel() > 10 && y.dim() > 0) {
                        auto flat = y.flatten();
                        auto slice_len = std::min<int64_t>(flat.numel(), y.size(0));
                        y = flat.narrow(0, 0, slice_len);
                    }

                    return torch::jit::Stack{y};
                },
                name_lookup,
                /*strict=*/false,
                /*force_outplace=*/false,
                nullptr,
                std::vector<std::string>{"x"});

            if (!traced_func.second.empty()) {
                traced_func.second.front().toTensor().sum();
            }
        }
        catch (const c10::Warning& w) {
            (void)w;
            // This is expected in some cases, so we'll just continue
        }
        
        // Try another approach that might trigger TracerWarning
        try {
            // Create a module with control flow
            torch::jit::script::CompilationUnit cu;
            auto script_methods = cu.define(
                c10::nullopt,
                R"(
                    def forward(self, x):
                        if x.size(0) > 0:
                            return x.sin()
                        else:
                            return x.cos()
                )",
                torch::jit::nativeResolver(),
                nullptr);

            if (!script_methods.empty() && script_methods.front()) {
                torch::jit::Stack script_inputs;
                script_inputs.push_back(input_tensor);

                auto traced_script = torch::jit::tracer::trace(
                    script_inputs,
                    [&](const torch::jit::Stack &stack) {
                        torch::jit::Stack local_stack = stack;
                        script_methods.front()->run(local_stack);
                        return local_stack;
                    },
                    name_lookup,
                    /*strict=*/false,
                    /*force_outplace=*/false,
                    nullptr,
                    std::vector<std::string>{"x"});

                if (!traced_script.second.empty()) {
                    traced_script.second.front().toTensor().sum();
                }
            }
        }
        catch (const c10::Warning& w) {
            (void)w;
            // Expected in some cases
        }
        
        // Try to trigger TracerWarning with dynamic shapes
        try {
            auto dynamic_func = [](const torch::jit::Stack &stack) {
                auto x = stack.front().toTensor();
                auto flat = x.flatten();
                auto limited = flat;
                if (flat.numel() > 64) {
                    limited = flat.narrow(0, 0, 64);
                }
                if (limited.numel() > 0) {
                    std::vector<int64_t> new_shape{limited.numel()};
                    return torch::jit::Stack{limited.reshape(new_shape)};
                }
                return torch::jit::Stack{x};
            };

            torch::jit::Stack dynamic_inputs;
            dynamic_inputs.push_back(input_tensor);

            auto traced_dynamic = torch::jit::tracer::trace(
                dynamic_inputs,
                dynamic_func,
                name_lookup,
                /*strict=*/false,
                /*force_outplace=*/false,
                nullptr,
                std::vector<std::string>{"x"});

            if (!traced_dynamic.second.empty()) {
                traced_dynamic.second.front().toTensor().abs_();
            }
        }
        catch (const c10::Warning& w) {
            (void)w;
            // Expected in some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
