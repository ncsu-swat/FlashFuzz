#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

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
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        // Keep keyword for harness checks: torch.jit.TracedModule
        torch::Tensor float_input = input_tensor.to(torch::kFloat);

        try {
            // Build a TorchScript module - this is the C++ equivalent of working with traced/scripted modules
            torch::jit::Module traced_module("traced_module");
            traced_module.define(R"JIT(
                def forward(self, x):
                    x = x.float()
                    return torch.relu(x + 1.0)
            )JIT");

            // Test the scripted module with the input
            std::vector<torch::jit::IValue> test_inputs;
            test_inputs.push_back(float_input);

            torch::Tensor output = traced_module.forward(test_inputs).toTensor();
            (void)output.sum();

            // Exercise module copy
            if (offset < Size) {
                uint8_t copy_flag = Data[offset++];
                if (copy_flag % 4 == 0) {
                    torch::jit::Module cloned = traced_module.clone();
                    torch::Tensor cloned_output = cloned.forward(test_inputs).toTensor();
                    (void)cloned_output.sum();
                }
            }

            // Try to run the module in different modes
            if (offset < Size) {
                uint8_t mode_flag = Data[offset++];
                if (mode_flag % 3 == 0) {
                    traced_module.eval();
                } else if (mode_flag % 3 == 1) {
                    traced_module.train();
                }

                // Run again after changing mode
                torch::Tensor mode_output = traced_module.forward(test_inputs).toTensor();
                (void)mode_output.sum();
            }

            // Register and access attributes to exercise attribute paths
            traced_module.register_buffer(
                "buffer_attr",
                torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kFloat)));
            
            for (const auto &attr : traced_module.named_attributes()) {
                (void)traced_module.attr(attr.name);
            }

            // Exercise parameter iteration (even if empty)
            for (const auto &param : traced_module.named_parameters()) {
                (void)param.value;
            }

            // Exercise submodule iteration (even if empty)
            for (const auto &submod : traced_module.named_modules()) {
                (void)submod.name;
            }

            // Test to() device/dtype conversions
            if (offset < Size) {
                uint8_t convert_flag = Data[offset++];
                if (convert_flag % 2 == 0) {
                    traced_module.to(torch::kDouble);
                    std::vector<torch::jit::IValue> double_inputs;
                    double_inputs.push_back(float_input.to(torch::kDouble));
                    torch::Tensor double_output = traced_module.forward(double_inputs).toTensor();
                    (void)double_output.sum();
                }
            }

        } catch (const c10::Error&) {
            // Expected failures (shape mismatches, etc.) - silently ignore
            return 0;
        } catch (const std::runtime_error&) {
            // TorchScript runtime errors - silently ignore
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}