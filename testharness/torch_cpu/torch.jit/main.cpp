#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

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
        
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Use remaining bytes to determine which JIT test to run
        uint8_t test_selector = 0;
        if (offset < Size) {
            test_selector = Data[offset++] % 6;
        }
        
        switch (test_selector) {
            case 0: {
                // Test: Define and run a simple module with identity
                torch::jit::Module module("test_module");
                module.define(R"JIT(
                    def forward(self, x):
                        return x
                )JIT");
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
                break;
            }
            case 1: {
                // Test: Basic arithmetic operations
                torch::jit::Module module("arith_module");
                module.define(R"JIT(
                    def forward(self, x):
                        return x + x * 2
                )JIT");
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
                break;
            }
            case 2: {
                // Test: Control flow with if statement
                torch::jit::Module module("control_module");
                module.define(R"JIT(
                    def forward(self, x):
                        if x.sum() > 0:
                            return x * 2
                        else:
                            return x * -1
                )JIT");
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
                break;
            }
            case 3: {
                // Test: Module with activation function
                torch::jit::Module module("activation_module");
                module.define(R"JIT(
                    def forward(self, x):
                        return torch.sigmoid(x)
                )JIT");
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
                break;
            }
            case 4: {
                // Test: Module with loop
                torch::jit::Module module("loop_module");
                module.define(R"JIT(
                    def forward(self, x):
                        result = x
                        for i in range(3):
                            result = result + x
                        return result
                )JIT");
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
                break;
            }
            case 5: {
                // Test: Module with multiple methods
                torch::jit::Module module("multi_method_module");
                module.define(R"JIT(
                    def helper(self, x):
                        return x.relu()
                    def forward(self, x):
                        y = self.helper(x)
                        return y.tanh()
                )JIT");
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
                break;
            }
        }
        
        // Test save to buffer and load (memory-based, not file-based)
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                torch::jit::Module module("buffer_test_module");
                module.define(R"JIT(
                    def forward(self, x):
                        return x * 2
                )JIT");
                
                // Save to stringstream (in-memory)
                std::ostringstream buffer;
                module.save(buffer);
                
                // Load from stringstream
                std::istringstream input_buffer(buffer.str());
                torch::jit::Module loaded_module = torch::jit::load(input_buffer);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = loaded_module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
            }
            catch (...) {
                // Serialization may fail for some configurations, ignore silently
            }
        }
        
        // Test cloning a module
        if (offset < Size && Data[offset++] % 3 == 0) {
            torch::jit::Module module("clone_test_module");
            module.define(R"JIT(
                def forward(self, x):
                    return x.abs()
            )JIT");
            
            torch::jit::Module cloned = module.clone();
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            auto output = cloned.forward(inputs);
            if (output.isTensor()) {
                torch::Tensor result = output.toTensor();
                (void)result.sum().item<float>();
            }
        }
        
        // Test module with parameters
        if (offset < Size && Data[offset++] % 3 == 0) {
            torch::jit::Module module("param_module");
            
            // Register a parameter
            auto weight = torch::randn({input_tensor.size(-1), input_tensor.size(-1)});
            module.register_parameter("weight", weight, false);
            
            module.define(R"JIT(
                def forward(self, x):
                    return torch.matmul(x, self.weight)
            )JIT");
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            try {
                auto output = module.forward(inputs);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum().item<float>();
                }
            }
            catch (...) {
                // Shape mismatch is expected for some inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}