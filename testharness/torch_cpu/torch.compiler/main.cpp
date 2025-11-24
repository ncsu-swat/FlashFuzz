#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/script.h> // For torch::jit::compile and run_method
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Target API keyword: torch.compiler
        // Create TorchScript code as string
        std::string simple_script = R"(
            def my_function(x):
                return torch.sin(x) + torch.cos(x)
        )";
        
        std::string complex_script = R"(
            def complex_function(x):
                a = torch.sin(x)
                b = torch.cos(x)
                c = torch.tanh(x)
                return a + b * c
        )";
        
        // Try different compilation modes based on available data
        if (offset < Size) {
            uint8_t compile_mode = Data[offset++] % 3;
            
            // Apply torch.jit.compile with different options
            try {
                if (compile_mode == 0) {
                    // Basic compilation
                    auto cu = torch::jit::compile(simple_script);
                    auto result = cu->run_method("my_function", input_tensor);
                    if (result.isTensor()) {
                        result.toTensor().sum();
                    }
                } else if (compile_mode == 1) {
                    // Compilation with optimization
                    auto cu = torch::jit::compile(simple_script);
                    auto result = cu->run_method("my_function", input_tensor);
                    if (result.isTensor()) {
                        result.toTensor().sum();
                    }
                } else {
                    // Try with complex function
                    auto cu = torch::jit::compile(complex_script);
                    auto result = cu->run_method("complex_function", input_tensor);
                    if (result.isTensor()) {
                        result.toTensor().sum();
                    }
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors but don't discard the input
                return 0;
            }
        } else {
            // Basic compilation if we don't have more data
            auto cu = torch::jit::compile(simple_script);
            auto result = cu->run_method("my_function", input_tensor);
            if (result.isTensor()) {
                result.toTensor().sum();
            }
        }
        
        // Try another compilation with a more complex function if we have more data
        if (offset + 2 < Size) {
            try {
                auto cu = torch::jit::compile(complex_script);
                auto result = cu->run_method("complex_function", input_tensor);
                if (result.isTensor()) {
                    result.toTensor().sum();
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors but don't discard the input
                return 0;
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
