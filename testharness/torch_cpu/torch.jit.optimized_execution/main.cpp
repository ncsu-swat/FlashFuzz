#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine if optimized execution should be enabled
        if (Size < 1) {
            return 0;
        }
        
        // Use first byte to determine if optimized execution should be enabled
        bool enable_optimized = Data[0] % 2 == 0;
        offset++;
        
        // Create a simple script module
        std::string script_code = R"(
def forward(self, x):
    return x + x
        )";
        
        torch::jit::Module module;
        try {
            auto cu = torch::jit::compile(script_code);
            module = torch::jit::Module(cu, "Module");
        } catch (const c10::Error& e) {
            // If compilation fails, try with a simpler module
            script_code = R"(
def forward(self, x):
    return x
            )";
            auto cu = torch::jit::compile(script_code);
            module = torch::jit::Module(cu, "Module");
        }
        
        // Set optimized execution mode
        torch::jit::setGraphExecutorOptimize(enable_optimized);
        
        // Create input tensor for the module
        torch::Tensor input;
        if (Size > offset) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if we don't have enough data
            input = torch::ones({1, 1});
        }
        
        // Create a vector of inputs for the module
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        // Execute the module with optimized execution setting
        torch::jit::IValue output = module.forward(inputs);
        
        // Try to get the tensor from the output
        if (output.isTensor()) {
            torch::Tensor result = output.toTensor();
            
            // Toggle optimization and run again to compare results
            torch::jit::setGraphExecutorOptimize(!enable_optimized);
            torch::jit::IValue output2 = module.forward(inputs);
            
            if (output2.isTensor()) {
                torch::Tensor result2 = output2.toTensor();
                
                // Check if results are the same regardless of optimization setting
                bool equal = torch::all(torch::eq(result, result2)).item<bool>();
                if (!equal) {
                    // Results differ between optimized and non-optimized execution
                    fuzzer_utils::saveDiffInput(Data, Size, fuzzer_utils::sanitizedTimestamp());
                }
            }
        }
        
        // Try with different optimization settings
        if (Size > offset + 1) {
            // Use next byte to determine different optimization flags
            uint8_t opt_flags = Data[offset++];
            
            // Test with different optimization settings
            bool fusion_enabled = (opt_flags & 0x01) != 0;
            torch::jit::setGraphExecutorOptimize(fusion_enabled);
            
            // Execute again with new settings
            torch::jit::IValue output3 = module.forward(inputs);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}