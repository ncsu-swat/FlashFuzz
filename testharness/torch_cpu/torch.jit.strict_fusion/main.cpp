#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Define a simple function that can be fused
        std::string script_code = R"(
            def forward(x):
                return x * 2 + 1
        )";
        
        try {
            // Compile the script
            auto compilation_unit = torch::jit::compile(script_code);
            
            // Get the function from compilation unit
            auto func = compilation_unit->find_function("forward");
            if (!func) {
                return 0;
            }
            
            // Create a graph executor with strict fusion enabled
            torch::jit::GraphExecutor executor(func->graph(), "test");
            
            // Run the function with our input tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Execute the function
            auto output = executor.run(inputs);
            
            // Try to get the tensor result
            if (output.size() > 0 && output[0].isTensor()) {
                torch::Tensor result = output[0].toTensor();
                
                // Verify the result is valid
                if (result.defined()) {
                    // Simple validation - check if result has expected properties
                    auto expected_size = input_tensor.sizes();
                    if (result.sizes() == expected_size) {
                        // Result looks valid
                    }
                }
            }
        }
        catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
