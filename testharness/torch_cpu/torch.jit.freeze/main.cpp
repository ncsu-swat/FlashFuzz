#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module using TorchScript compilation
        // We need to compile a module from source to get a valid scriptable module
        std::string module_source = R"(
            def forward(self, x):
                return x * 2 + 1
        )";
        
        torch::jit::Module module("TestModule");
        
        // Register a parameter to make freezing more interesting
        torch::Tensor weight = torch::ones({1});
        module.register_parameter("weight", weight, false);
        
        // Define the forward method
        module.define(R"(
            def forward(self, x):
                return x * self.weight + 1
        )");
        
        // Put module in eval mode (required for freeze)
        module.eval();
        
        // Get fuzzer-controlled options
        bool preserve_parameters = false;
        if (offset < Size) {
            preserve_parameters = Data[offset++] & 0x1;
        }
        
        // Call torch::jit::freeze with the module
        // freeze() returns a new frozen module
        try {
            torch::jit::Module frozen_module;
            if (preserve_parameters) {
                // Preserve specific attributes
                std::vector<std::string> preserved_attrs = {"weight"};
                frozen_module = torch::jit::freeze(module, preserved_attrs);
            } else {
                // Freeze all parameters
                frozen_module = torch::jit::freeze(module);
            }
            
            // Run forward on the frozen module to verify it works
            auto result = frozen_module.forward({input_tensor});
            
            // Additional fuzzing: try different input shapes
            if (offset + 2 < Size) {
                int dim0 = (Data[offset++] % 8) + 1;
                int dim1 = (Data[offset++] % 8) + 1;
                torch::Tensor another_input = torch::randn({dim0, dim1});
                try {
                    auto result2 = frozen_module.forward({another_input});
                } catch (const c10::Error& e) {
                    // Shape mismatch is expected, continue
                }
            }
            
        } catch (const c10::Error& e) {
            // Expected errors from freeze (e.g., module not in eval mode)
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