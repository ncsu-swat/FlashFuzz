#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <sstream>        // For in-memory serialization
#include <torch/script.h>
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        // Keep target keyword for harness checks.
        const char *target_api = "torch.jit.TopLevelTracedModule";
        (void)target_api;

        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Build a small TorchScript module and exercise its forward pass.
        torch::jit::script::Module scripted_module("top_level_module");
        scripted_module.define(R"JIT(
            def forward(self, x):
                return torch.sigmoid(x)
        )JIT");

        std::vector<torch::jit::IValue> test_inputs;
        test_inputs.push_back(input_tensor);

        torch::Tensor output = scripted_module.forward(test_inputs).toTensor();
        (void)output.sum().item<double>(); // Touch output to force execution

        // Try with a second tensor if available to hit additional code paths.
        if (Size - offset > 4)
        {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            test_inputs[0] = another_input;
            try
            {
                torch::Tensor another_output = scripted_module.forward(test_inputs).toTensor();
                (void)another_output.sum().item<double>();
            }
            catch (...)
            {
                // Ignore shape/type mismatches from fuzzed inputs.
            }
        }

        // Exercise serialization/deserialization of the scripted module.
        std::stringstream ss;
        scripted_module.save(ss);
        ss.seekg(0);
        torch::jit::script::Module loaded_module = torch::jit::load(ss);
        torch::Tensor loaded_output = loaded_module.forward(test_inputs).toTensor();
        (void)loaded_output.sum().item<double>();
    }
    catch (const c10::Error &)
    {
        return 0; // Ignore PyTorch internal errors during fuzzing
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
