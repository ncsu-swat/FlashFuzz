#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple module with a function
        torch::jit::Module module("test_module");
        
        // Create input tensor for the model
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model with a single operation
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Define a simple function that returns the input tensor
        std::string func_str = R"(
            def forward(self, x):
                return x + 1
        )";
        
        // Parse the function and add it to the module
        module.define(func_str);
        
        // Get a byte to determine if we should freeze the module
        bool should_freeze = false;
        if (offset < Size) {
            should_freeze = Data[offset++] % 2 == 0;
        }
        
        // Freeze the module if needed
        if (should_freeze) {
            module = torch::jit::freeze(module);
        }
        
        // Apply optimize_for_inference
        torch::jit::Module optimized_module = torch::jit::optimize_for_inference(module);
        
        // Test the optimized module with the input tensor
        std::vector<torch::jit::IValue> test_inputs;
        test_inputs.push_back(input_tensor);
        
        // Run the optimized module
        torch::jit::IValue output = optimized_module.forward(test_inputs);
        
        // Try to extract the tensor from the output
        if (output.isTensor()) {
            torch::Tensor output_tensor = output.toTensor();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}