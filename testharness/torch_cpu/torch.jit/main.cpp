#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h> // For torch::jit functionality

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module to test JIT functionality
        std::string module_code;
        
        // Use remaining bytes to determine which JIT test to run
        if (offset < Size) {
            uint8_t test_selector = Data[offset++] % 5;
            
            switch (test_selector) {
                case 0: {
                    // Simple identity function
                    module_code = R"(
                        def forward(self, x):
                            return x
                    )";
                    break;
                }
                case 1: {
                    // Basic arithmetic
                    module_code = R"(
                        def forward(self, x):
                            return x + x
                    )";
                    break;
                }
                case 2: {
                    // Control flow
                    module_code = R"(
                        def forward(self, x):
                            if x.sum() > 0:
                                return x * 2
                            else:
                                return x * -1
                    )";
                    break;
                }
                case 3: {
                    // More complex operations
                    module_code = R"(
                        def forward(self, x):
                            return torch.nn.functional.relu(x)
                    )";
                    break;
                }
                case 4: {
                    // Try with loops
                    module_code = R"(
                        def forward(self, x):
                            result = x
                            for _ in range(3):
                                result = result + x
                            return result
                    )";
                    break;
                }
            }
        } else {
            // Default module if we've consumed all data
            module_code = R"(
                def forward(self, x):
                    return x
            )";
        }
        
        // Try different JIT compilation methods
        if (offset < Size) {
            uint8_t jit_method = Data[offset++] % 3;
            
            switch (jit_method) {
                case 0: {
                    // Method 1: Compile a TorchScript function from a string
                    auto compilation_unit = torch::jit::compile(module_code);
                    auto function = compilation_unit->find_function("forward");
                    if (function) {
                        std::vector<torch::jit::IValue> inputs;
                        inputs.push_back(input_tensor);
                        torch::jit::IValue output = function->operator()(inputs);
                        torch::Tensor result = output.toTensor();
                    }
                    break;
                }
                case 1: {
                    // Method 2: Trace a C++ function
                    auto lambda_func = [](torch::Tensor x) -> torch::Tensor {
                        return x.sigmoid();
                    };
                    
                    torch::jit::Module traced_module = torch::jit::trace(lambda_func, {input_tensor});
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    torch::jit::IValue output = traced_module.forward(inputs);
                    torch::Tensor result = output.toTensor();
                    break;
                }
                case 2: {
                    // Method 3: Create a simple traced module
                    auto simple_func = [](torch::Tensor x) -> torch::Tensor {
                        return x.tanh();
                    };
                    
                    torch::jit::Module scripted_module = torch::jit::trace(simple_func, {input_tensor});
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    torch::jit::IValue output = scripted_module.forward(inputs);
                    torch::Tensor result = output.toTensor();
                    break;
                }
            }
        } else {
            // Default JIT method if we've consumed all data
            auto compilation_unit = torch::jit::compile(module_code);
            auto function = compilation_unit->find_function("forward");
            if (function) {
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                torch::jit::IValue output = function->operator()(inputs);
                torch::Tensor result = output.toTensor();
            }
        }
        
        // Test serialization and deserialization if we have more data
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create a simple traced module
            auto simple_func = [](torch::Tensor x) -> torch::Tensor {
                return x * 2;
            };
            
            torch::jit::Module scripted_module = torch::jit::trace(simple_func, {input_tensor});
            
            // Save to a temporary file
            scripted_module.save("temp_module.pt");
            
            // Load it back
            torch::jit::Module loaded_module = torch::jit::load("temp_module.pt");
            
            // Run the loaded module
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            torch::jit::IValue output = loaded_module.forward(inputs);
            torch::Tensor result = output.toTensor();
        }
    }
    catch (const c10::Error &e)
    {
        return 0;
    }
    catch (const std::exception &e)
    {
        return 0;
    }
    return 0;
}