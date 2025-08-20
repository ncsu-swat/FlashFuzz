#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create another tensor if there's data left
        torch::Tensor input2;
        if (offset + 2 < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input2 = torch::ones_like(input1);
        }
        
        // Select which method to test based on remaining data
        uint8_t method_selector = 0;
        if (offset < Size) {
            method_selector = Data[offset++] % 4;
        }
        
        // Define the methods using torch.jit.script
        try {
            torch::jit::script::Module script_module("test_module");
            
            // Test different methods based on the selector
            switch (method_selector) {
                case 0: {
                    script_module.define(R"(
                        def forward(self, x):
                            return x + 1
                    )");
                    auto output = script_module.forward({input1});
                    break;
                }
                case 1: {
                    script_module.define(R"(
                        def add(self, x, y):
                            return x + y
                    )");
                    auto output = script_module.run_method("add", input1, input2);
                    break;
                }
                case 2: {
                    script_module.define(R"(
                        def multiply(self, x, y):
                            return x * y
                    )");
                    auto output = script_module.run_method("multiply", input1, input2);
                    break;
                }
                case 3: {
                    script_module.define(R"(
                        def identity(self, x):
                            return x
                    )");
                    auto output = script_module.run_method("identity", input1);
                    break;
                }
            }
            
            // Test with more complex script methods
            if (offset < Size && Data[offset] % 2 == 0) {
                torch::jit::script::Module complex_module("complex_module");
                complex_module.define(R"(
                    def complex_method(self, x, y):
                        z = x + y
                        if z.sum() > 0:
                            return z * 2
                        else:
                            return z - 1
                )");
                auto output = complex_module.run_method("complex_method", input1, input2);
            }
            
            // Test with control flow
            if (offset < Size && Data[offset] % 3 == 0) {
                torch::jit::script::Module control_module("control_module");
                control_module.define(R"(
                    def control_flow(self, x):
                        result = x
                        for i in range(3):
                            result = result + i
                        return result
                )");
                auto output = control_module.run_method("control_flow", input1);
            }
            
            // Test with nested function calls
            if (offset < Size && Data[offset] % 5 == 0) {
                torch::jit::script::Module nested_module("nested_module");
                nested_module.define(R"(
                    def helper(self, x):
                        return x * 2
                        
                    def nested_calls(self, x):
                        return self.helper(x) + self.helper(x + 1)
                )");
                auto output = nested_module.run_method("nested_calls", input1);
            }
        }
        catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}