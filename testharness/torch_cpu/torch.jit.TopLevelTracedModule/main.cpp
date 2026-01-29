#include "fuzzer_utils.h"
#include <iostream>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        const char *target_api = "torch.jit.TopLevelTracedModule";
        (void)target_api;

        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor from fuzz data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Create a scripted module and exercise it
        torch::jit::script::Module scripted_module("top_level_module");
        scripted_module.define(R"JIT(
            def forward(self, x):
                return torch.sigmoid(x)
        )JIT");

        std::vector<torch::jit::IValue> test_inputs;
        test_inputs.push_back(input_tensor);

        torch::Tensor output = scripted_module.forward(test_inputs).toTensor();
        (void)output.sum().item<double>();

        // Exercise serialization/deserialization of the scripted module
        std::stringstream ss;
        scripted_module.save(ss);
        ss.seekg(0);
        torch::jit::script::Module loaded_module = torch::jit::load(ss);
        torch::Tensor loaded_output = loaded_module.forward(test_inputs).toTensor();
        (void)loaded_output.sum().item<double>();

        // Create a more complex scripted module
        torch::jit::script::Module complex_module("complex_module");
        complex_module.define(R"JIT(
            def forward(self, x):
                y = torch.relu(x)
                z = torch.tanh(y)
                return z + x
        )JIT");
        
        torch::Tensor complex_output = complex_module.forward(test_inputs).toTensor();
        (void)complex_output.sum().item<double>();

        // Serialize and load the complex module
        std::stringstream ss2;
        complex_module.save(ss2);
        ss2.seekg(0);
        torch::jit::script::Module loaded_complex = torch::jit::load(ss2);
        torch::Tensor loaded_complex_output = loaded_complex.forward(test_inputs).toTensor();
        (void)loaded_complex_output.sum().item<double>();

        // Try with another fuzzed tensor
        if (Size - offset > 4)
        {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            std::vector<torch::jit::IValue> another_inputs;
            another_inputs.push_back(another_input);
            try
            {
                torch::Tensor another_output = scripted_module.forward(another_inputs).toTensor();
                (void)another_output.sum().item<double>();
            }
            catch (...)
            {
                // Ignore shape/type mismatches from fuzzed inputs
            }
        }

        // Exercise module introspection methods
        (void)scripted_module.dump_to_str(true, false, false);
        
        // Exercise copy functionality
        torch::jit::script::Module cloned = scripted_module.clone();
        torch::Tensor cloned_output = cloned.forward(test_inputs).toTensor();
        (void)cloned_output.sum().item<double>();

        // Exercise module with parameters
        torch::jit::script::Module param_module("param_module");
        param_module.register_parameter("weight", torch::randn({4, 4}), false);
        param_module.define(R"JIT(
            def forward(self, x):
                w = self.weight
                # Handle broadcasting - use element-wise operations
                return torch.sigmoid(x * w.sum())
        )JIT");
        
        try {
            torch::Tensor param_input = torch::randn({2, 4});
            std::vector<torch::jit::IValue> param_inputs;
            param_inputs.push_back(param_input);
            torch::Tensor param_output = param_module.forward(param_inputs).toTensor();
            (void)param_output.sum().item<double>();
        }
        catch (...) {
            // Parameter operations may fail with certain shapes
        }

        // Exercise module attributes
        torch::jit::script::Module attr_module("attr_module");
        attr_module.register_attribute("scale", torch::jit::FloatType::get(), torch::jit::IValue(2.0));
        attr_module.define(R"JIT(
            def forward(self, x):
                return x * self.scale
        )JIT");
        
        torch::Tensor attr_output = attr_module.forward(test_inputs).toTensor();
        (void)attr_output.sum().item<double>();

        // Test module hierarchy
        torch::jit::script::Module parent_module("parent");
        torch::jit::script::Module child_module("child");
        child_module.define(R"JIT(
            def forward(self, x):
                return torch.relu(x)
        )JIT");
        parent_module.register_module("child", child_module);
        parent_module.define(R"JIT(
            def forward(self, x):
                return self.child.forward(x) + x
        )JIT");
        
        torch::Tensor parent_output = parent_module.forward(test_inputs).toTensor();
        (void)parent_output.sum().item<double>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}