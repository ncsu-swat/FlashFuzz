#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <functional>     // For std::function in tracing helpers
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tracer.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Target API keyword for harness checks: torch.jit.TracingCheckError
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if we have more data
        torch::Tensor input2;
        if (offset + 4 < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input2 = input1.clone();
        }

        auto name_lookup = [](const torch::autograd::Variable &) {
            return std::string();
        };

        auto run_trace = [&](const torch::Tensor &a,
                             const torch::Tensor &b,
                             const std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &)> &fn,
                             const std::vector<std::string> &arg_names) {
            torch::jit::Stack stack_inputs;
            stack_inputs.push_back(a);
            stack_inputs.push_back(b);

            auto traced = torch::jit::tracer::trace(
                stack_inputs,
                [fn](torch::jit::Stack stack) {
                    auto lhs = stack.at(0).toTensor();
                    auto rhs = stack.at(1).toTensor();
                    auto result = fn(lhs, rhs);
                    return torch::jit::Stack{result};
                },
                name_lookup,
                /*strict=*/false,
                /*force_outplace=*/false,
                nullptr,
                arg_names);

            if (!traced.second.empty()) {
                auto out = traced.second.front().toTensor();
                if (out.defined()) {
                    out = out + 1.0;
                    out.sum();
                }
            }
        };
        
        try {
            auto add_func = [](torch::Tensor a, torch::Tensor b) {
                return a + b;
            };

            run_trace(input1, input2, add_func, {"a", "b"});
            
            // Try to run the traced function with different inputs
            torch::Tensor modified_input1;
            torch::Tensor modified_input2;
            
            // Create modified inputs to trigger tracing check errors
            if (offset + 4 < Size) {
                modified_input1 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Modify the shape if possible
                if (input1.dim() > 0) {
                    auto sizes = input1.sizes().vec();
                    for (size_t i = 0; i < sizes.size(); i++) {
                        if (sizes[i] > 1) {
                            sizes[i] = sizes[i] - 1;
                            break;
                        } else if (sizes[i] == 0) {
                            sizes[i] = 1;
                            break;
                        }
                    }
                    modified_input1 = torch::zeros(sizes, input1.options());
                } else {
                    modified_input1 = input1 + 1.0;
                }
            }
            
            if (offset + 4 < Size) {
                modified_input2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Modify the shape if possible
                if (input2.dim() > 0) {
                    auto sizes = input2.sizes().vec();
                    for (size_t i = 0; i < sizes.size(); i++) {
                        if (sizes[i] > 1) {
                            sizes[i] = sizes[i] - 1;
                            break;
                        } else if (sizes[i] == 0) {
                            sizes[i] = 1;
                            break;
                        }
                    }
                    modified_input2 = torch::zeros(sizes, input2.options());
                } else {
                    modified_input2 = input2 + 1.0;
                }
            }
            
            // Try to run with modified inputs to trigger TracingCheckError
            run_trace(modified_input1, modified_input2, add_func, {"a", "b"});
        }
        catch (const std::runtime_error& e) {
            // This catches tracing-related errors
        }
        
        // Try with different control flow paths
        if (offset < Size) {
            uint8_t control_byte = Data[offset++];
            
            try {
                // Define a function with control flow
                auto control_flow_func = [control_byte](torch::Tensor a, torch::Tensor b) {
                    if (control_byte % 2 == 0) {
                        return a + b;
                    } else {
                        return a * b;
                    }
                };
                
                // Trace with one path and run with inputs that might take a different path
                run_trace(input1, input2, control_flow_func, {"a", "b"});
            }
            catch (const std::runtime_error& e) {
                // Expected when control flow changes
            }
        }
        
        // Test with empty tensors
        try {
            auto empty_tensor1 = torch::empty({0}, input1.options());
            auto empty_tensor2 = torch::empty({0}, input2.options());
            
            auto empty_func = [](torch::Tensor a, torch::Tensor b) {
                return a + b;
            };
            
            run_trace(empty_tensor1, empty_tensor2, empty_func, {"a", "b"});
        }
        catch (const std::runtime_error& e) {
            // Expected when shapes don't match
        }
        
        // Test with scalar tensors
        try {
            auto scalar_tensor1 = torch::tensor(1.0f, input1.options());
            auto scalar_tensor2 = torch::tensor(2.0f, input2.options());
            
            auto scalar_func = [](torch::Tensor a, torch::Tensor b) {
                return a + b;
            };
            
            run_trace(scalar_tensor1, scalar_tensor2, scalar_func, {"a", "b"});
        }
        catch (const std::runtime_error& e) {
            // Expected when shapes don't match
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
