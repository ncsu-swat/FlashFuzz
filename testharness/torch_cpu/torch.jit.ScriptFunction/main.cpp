#include "fuzzer_utils.h"                          // General fuzzing utilities
#include <torch/csrc/jit/api/compilation_unit.h>   // StrongFunctionPtr (ScriptFunction binding)
#include <torch/script.h>                          // torch::jit::compile and Stack/IValue
#include <iostream>                                // For cerr/cout

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

        // Need at least a few bytes to create a tensor and define a script function
        if (Size < 4) {
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Define a simple script function
        std::string script_code;

        // Use remaining bytes to determine which function to test
        if (offset < Size) {
            uint8_t func_selector = Data[offset++] % 8;
            
            switch (func_selector) {
                case 0:
                    script_code = "def forward(x):\n  return x.sin()";
                    break;
                case 1:
                    script_code = "def forward(x):\n  return x.abs() + 1";
                    break;
                case 2:
                    script_code = "def forward(x):\n  return x * x";
                    break;
                case 3:
                    script_code = "def forward(x):\n  return x.sum()";
                    break;
                case 4:
                    script_code = "def forward(x):\n  return x.exp()";
                    break;
                case 5:
                    script_code = "def forward(x):\n  return x.cos()";
                    break;
                case 6:
                    script_code = "def forward(x):\n  return x.tanh()";
                    break;
                case 7:
                default:
                    script_code = "def forward(x):\n  return x + 1";
                    break;
            }
        } else {
            script_code = "def forward(x):\n  return x + 1";
        }

        // Compile the script function
        std::shared_ptr<torch::jit::CompilationUnit> cu;
        try {
            cu = torch::jit::compile(script_code);
        } catch (...) {
            // If compilation fails, try a simpler function
            script_code = "def forward(x):\n  return x";
            cu = torch::jit::compile(script_code);
        }

        // Get the forward function from the compilation unit
        torch::jit::Function& forward_func = cu->get_function("forward");

        // StrongFunctionPtr backs the Python-exposed torch.jit.ScriptFunction
        torch::jit::StrongFunctionPtr script_function(cu, &forward_func);

        // Prepare call stack and invoke the ScriptFunction
        // The function modifies the stack in place - input goes in, output comes out
        torch::jit::Stack stack;
        stack.emplace_back(input_tensor);
        
        // Call the function - this modifies stack in place
        script_function.function_->run(stack);

        // Extract result from the stack
        if (!stack.empty()) {
            torch::jit::IValue output = stack.front();
            
            if (output.isTensor()) {
                torch::Tensor result = output.toTensor();

                // Perform some operation on the result to ensure it's used
                if (result.defined() && result.numel() > 0) {
                    auto sum = result.sum().item<float>();
                    (void)sum; // Suppress unused variable warning
                }
            }
        }

        // Test calling the function again with a different input
        if (offset < Size) {
            torch::Tensor input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::jit::Stack stack2;
            stack2.emplace_back(input_tensor2);
            script_function.function_->run(stack2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}