#include "fuzzer_utils.h"
#include <iostream>
#include <torch/csrc/jit/serialization/export.h>
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
        
        // Create a tensor from the fuzzer data
        torch::Tensor input_tensor;
        if (Size > 0) {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure tensor is at least 2D for matrix operations
            if (input_tensor.dim() < 2) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        } else {
            input_tensor = torch::ones({2, 2});
        }
        
        // Test 1: Simple module with basic operations
        {
            torch::jit::Module module("test_module");
            
            std::string script_code = R"(
def forward(self, x: Tensor) -> Tensor:
    y = x + 1
    z = torch.relu(y)
    return z * 2
)";
            
            try {
                module.define(script_code);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                module.forward(inputs);
                
                auto op_names = torch::jit::export_opnames(module);
                // Use the result to prevent optimization
                volatile size_t count = op_names.size();
                (void)count;
            } catch (...) {
                // Script compilation or execution may fail with certain inputs
            }
        }
        
        // Test 2: Module with more operations based on fuzzer data
        if (Size > 4) {
            torch::jit::Module module2("module2");
            
            std::string script_code2 = R"(
def forward(self, x: Tensor) -> Tensor:
    y = torch.abs(x)
    z = torch.sum(y)
    return z
)";
            
            try {
                module2.define(script_code2);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                module2.forward(inputs);
                
                auto op_names = torch::jit::export_opnames(module2);
                volatile size_t count = op_names.size();
                (void)count;
            } catch (...) {
                // May fail with certain tensor configurations
            }
        }
        
        // Test 3: Module with conditional operations
        if (Size > 8) {
            torch::jit::Module module3("module3");
            
            std::string script_code3 = R"(
def forward(self, x: Tensor) -> Tensor:
    if x.numel() > 0:
        y = torch.neg(x)
        z = torch.exp(y.clamp(-10, 10))
    else:
        z = torch.zeros(1)
    return z
)";
            
            try {
                module3.define(script_code3);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                module3.forward(inputs);
                
                auto op_names = torch::jit::export_opnames(module3);
                volatile size_t count = op_names.size();
                (void)count;
            } catch (...) {
                // May fail
            }
        }
        
        // Test 4: Empty module (valid case)
        {
            torch::jit::Module empty_module("empty_module");
            auto empty_op_names = torch::jit::export_opnames(empty_module);
            // Empty result is valid for empty module
            volatile size_t count = empty_op_names.size();
            (void)count;
        }
        
        // Test 5: Module with multiple methods
        if (Size > 12) {
            torch::jit::Module multi_module("multi_module");
            
            std::string multi_script = R"(
def forward(self, x: Tensor) -> Tensor:
    return self.helper(x) + 1

def helper(self, x: Tensor) -> Tensor:
    return torch.sigmoid(x)
)";
            
            try {
                multi_module.define(multi_script);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                multi_module.forward(inputs);
                
                auto op_names = torch::jit::export_opnames(multi_module);
                volatile size_t count = op_names.size();
                (void)count;
            } catch (...) {
                // May fail
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