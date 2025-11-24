#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/jit/serialization/export.h>
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Create a simple module with a method
        torch::jit::Module module("test_module");
        
        // Create a tensor from the fuzzer data
        torch::Tensor input_tensor;
        if (Size > 0) {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input_tensor = torch::ones({1, 1});
        }
        
        // Create a simple script function that uses various operations
        std::string script_code = R"(
            def forward(self, x):
                y = x + 1
                z = torch.relu(y)
                return z * 2
        )";
        
        // Define the module with the script
        module.define(script_code);
        
        // Create inputs for the module
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Run the module to ensure operations are recorded
        module.forward(inputs);
        
        // Test export_opnames with different parameters
        // Call export_opnames with the scripted module
        auto op_names = torch::jit::export_opnames(module);
        
        // Verify the result is not empty
        if (op_names.empty()) {
            throw std::runtime_error("export_opnames returned empty list");
        }
        
        // Try with a more complex module if we have enough data
        if (Size > 2) {
            torch::jit::Module complex_module("complex_module");
            
            // Create a more complex script with various operations
            std::string complex_script = R"(
                def forward(self, x):
                    y = torch.nn.functional.relu(x)
                    z = torch.matmul(y, y.t())
                    a = torch.mean(z, dim=0)
                    b = torch.cat([a, a], dim=0)
                    return torch.softmax(b, dim=0)
            )";
            
            // Define the complex module
            complex_module.define(complex_script);
            
            // Run the module to ensure operations are recorded
            complex_module.forward(inputs);
            
            // Test export_opnames with the complex module
            auto op_names = torch::jit::export_opnames(complex_module);
        }
        
        // Test with an empty module
        torch::jit::Module empty_module("empty_module");
        auto empty_op_names = torch::jit::export_opnames(empty_module);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
