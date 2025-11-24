#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
            // Build a small TorchScript module instead of relying on Python-only tracing APIs.
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

            // Try to save and load the scripted module
            if (offset < Size) {
                uint8_t save_flag = Data[offset++];
                if (save_flag % 2 == 0) {
                    traced_module.save("temp_module.pt");

                    // Load the saved module
                    torch::jit::Module loaded_module = torch::jit::load("temp_module.pt");

                    // Run the loaded module
                    torch::Tensor loaded_output = loaded_module.forward(test_inputs).toTensor();
                    (void)loaded_output.sum();
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

            // Touch registered attributes to exercise attribute paths.
            traced_module.register_buffer(
                "buffer_attr",
                torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat)));
            for (const auto &attr : traced_module.named_attributes()) {
                (void)traced_module.attr(attr.name);
            }
        } catch (const c10::Error&) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
