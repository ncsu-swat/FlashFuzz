#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/frontend/resolver.h>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use in our JIT code
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a byte to determine the number of functions to add
        uint8_t num_functions = 1;
        if (offset < Size) {
            num_functions = (Data[offset++] % 3) + 1; // 1-3 functions
        }
        
        // Create a CompilationUnit
        torch::jit::CompilationUnit cu;
        
        // Track which functions were successfully defined
        std::vector<std::string> defined_functions;
        
        // Define some TorchScript functions based on the input data
        for (uint8_t i = 0; i < num_functions; i++) {
            // Choose function type based on input data
            uint8_t func_type = 0;
            if (offset < Size) {
                func_type = Data[offset++] % 6;
            }
            
            std::string func_name = "test_func_" + std::to_string(i);
            std::string source;
            
            switch (func_type) {
                case 0:
                    // Identity function
                    source = "def " + func_name + "(x):\n    return x\n";
                    break;
                case 1:
                    // Add a constant
                    source = "def " + func_name + "(x):\n    return x + 1.0\n";
                    break;
                case 2:
                    // Multiply by a constant
                    source = "def " + func_name + "(x):\n    return x * 2.0\n";
                    break;
                case 3:
                    // Apply ReLU
                    source = "def " + func_name + "(x):\n    return x.relu()\n";
                    break;
                case 4:
                    // More complex function
                    source = "def " + func_name + "(x):\n    y = x * x\n    z = y + x\n    return z.tanh()\n";
                    break;
                case 5:
                    // Negation and abs
                    source = "def " + func_name + "(x):\n    return (-x).abs()\n";
                    break;
            }
            
            try {
                cu.define(
                    c10::nullopt,  // prefix
                    source,        // source code
                    torch::jit::nativeResolver(),  // resolver
                    nullptr        // self
                );
                defined_functions.push_back(func_name);
            } catch (...) {
                // Function definition might fail, continue testing
            }
        }
        
        // Try to get and call functions from the CompilationUnit
        if (!defined_functions.empty()) {
            // Choose which function to call
            uint8_t func_idx = 0;
            if (offset < Size) {
                func_idx = Data[offset++] % defined_functions.size();
            }
            
            std::string func_to_call = defined_functions[func_idx];
            
            try {
                // Get the function
                torch::jit::Function& method = cu.get_function(func_to_call);
                
                // Create a list of inputs for the function
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Call the function
                auto output = method(inputs);
                
                // Try to extract the tensor from the output
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    
                    // Perform some operation on the result to ensure it's used
                    auto sum = result.sum();
                    (void)sum;
                }
            } catch (...) {
                // Function call might fail due to tensor incompatibility
            }
        }
        
        // Test getting all functions
        try {
            auto functions = cu.get_functions();
            (void)functions;
        } catch (...) {
            // Ignore errors
        }
        
        // Try to define an invalid function to test error handling
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                cu.define(
                    c10::nullopt,
                    "def invalid_func(x):\n    invalid syntax here\n",
                    torch::jit::nativeResolver(),
                    nullptr
                );
            } catch (...) {
                // Expected to fail
            }
        }
        
        // Try to get a non-existent function
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::jit::Function& non_existent = cu.get_function("non_existent_function");
                (void)non_existent;
            } catch (...) {
                // Expected to fail
            }
        }
        
        // Test find_function which returns nullptr instead of throwing
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                auto* found = cu.find_function("test_func_0");
                if (found != nullptr) {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor.clone());
                    auto result = (*found)(inputs);
                    (void)result;
                }
            } catch (...) {
                // Ignore errors
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