#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

class TestModule : public torch::nn::Module {
public:
    TestModule() {}

    torch::Tensor forward(torch::Tensor x) {
        return x + 1;
    }
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module to export
        auto module = std::make_shared<TestModule>();
        
        // Try different export scenarios based on remaining data
        if (offset < Size) {
            uint8_t export_option = Data[offset++] % 3;
            
            switch (export_option) {
                case 0: {
                    // Basic export
                    torch::jit::script::Module scripted_module = torch::jit::script::Module();
                    torch::jit::script::Module submodule = torch::jit::script::Module();
                    
                    // Test the exported module
                    auto result = module->forward(input_tensor);
                    break;
                }
                case 1: {
                    // Export with method
                    auto script_module = torch::jit::script::Module();
                    torch::jit::script::Module submodule = torch::jit::script::Module();
                    
                    // Define a method to export
                    auto method = [&](torch::Tensor x) -> torch::Tensor {
                        return x * 2;
                    };
                    
                    // Try to use the exported module
                    auto result = method(input_tensor);
                    break;
                }
                case 2: {
                    // Export with attributes
                    auto script_module = torch::jit::script::Module();
                    
                    // Register various attributes based on tensor properties
                    if (input_tensor.dim() > 0) {
                        script_module.register_attribute("tensor_dim", torch::jit::IntType::get(), input_tensor.dim());
                    }
                    
                    if (input_tensor.numel() > 0) {
                        script_module.register_attribute("tensor_numel", torch::jit::IntType::get(), input_tensor.numel());
                    }
                    
                    // Register the tensor itself as an attribute if possible
                    try {
                        script_module.register_attribute("input_tensor", torch::jit::TensorType::get(), input_tensor);
                    } catch (...) {
                        // Ignore errors when registering tensor
                    }
                    
                    // Try to access the registered attributes
                    if (script_module.hasattr("tensor_dim")) {
                        int64_t dim = script_module.attr("tensor_dim").toInt();
                    }
                    
                    if (script_module.hasattr("input_tensor")) {
                        torch::Tensor t = script_module.attr("input_tensor").toTensor();
                    }
                    break;
                }
            }
        } else {
            // Default case if we don't have enough data for options
            auto module_script = torch::jit::script::Module();
            
            // Test the module
            auto result = module->forward(input_tensor);
        }
        
        // Try to save and load the module if we have enough data
        if (offset < Size && Data[offset] % 2 == 0) {
            auto module_script = torch::jit::script::Module();
            
            // Save to a temporary file
            module_script.save("temp_module.pt");
            
            // Load the saved module
            auto loaded_module = torch::jit::load("temp_module.pt");
            
            // Test the loaded module with our input tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            try {
                auto output = loaded_module.forward(inputs);
            } catch (...) {
                // Ignore errors when running the loaded module
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}