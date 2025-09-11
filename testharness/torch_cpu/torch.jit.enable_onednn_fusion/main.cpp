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
        
        // Need at least 1 byte to determine if we should enable or disable oneDNN fusion
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine if we should enable or disable oneDNN fusion
        bool enable_fusion = (Data[offset++] % 2 == 0);
        
        // Call the setGraphExecutorOptimize function to control optimizations
        torch::jit::setGraphExecutorOptimize(enable_fusion);
        
        // Create a simple model to test if the fusion setting works
        if (Size >= offset + 2) {
            // Create a tensor to use in a JIT model
            torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a simple JIT script module that could potentially use oneDNN fusion
            std::string script_code = R"(
                def forward(self, x):
                    return x + x
            )";
            
            try {
                // Compile the script
                auto module = torch::jit::compile(script_code);
                
                // Run the module with our input tensor
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Execute the model using run_method
                auto output = module->run_method("forward", inputs);
                
                // Try to access the output tensor to ensure computation completed
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                }
            } catch (const c10::Error& e) {
                // JIT-specific errors are expected and should not terminate fuzzing
                return 0;
            }
        }
        
        // Test toggling the fusion setting
        torch::jit::setGraphExecutorOptimize(!enable_fusion);
        
        // Test with a different tensor if we have more data
        if (Size >= offset + 2 && offset < Size) {
            try {
                torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Create another simple module
                std::string script_code = R"(
                    def forward(self, x):
                        return x * 2
                )";
                
                auto module = torch::jit::compile(script_code);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(another_tensor);
                
                auto output = module->run_method("forward", inputs);
                
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                }
            } catch (const c10::Error& e) {
                // JIT-specific errors are expected
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
