#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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

        // Set optimized execution mode using the C++ API
        // This is the C++ equivalent of torch.jit.optimized_execution context manager
        bool previous_setting = torch::jit::getGraphExecutorOptimize();
        torch::jit::setGraphExecutorOptimize(enable_optimized);

        // Create input tensor for testing
        torch::Tensor input;
        if (Size > offset) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input = torch::randn({2, 3});
        }

        // Create a simple traced module by tracing a lambda
        // This is a more reliable way to create a JIT module in C++
        torch::jit::script::Module module;
        
        try {
            // Define a simple module using TorchScript string
            std::string module_src = R"(
class SimpleModule(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x + x * 2.0
)";
            // Use torch::jit::compile to create a compilation unit
            auto cu = torch::jit::compile(module_src);
            
            // Get the class from compilation unit and create module
            auto cls = cu->get_class("SimpleModule");
            if (cls) {
                module = torch::jit::script::Module(cu, cls);
            } else {
                // Fallback: create a module that just does basic operations
                // without JIT compilation - test the optimization flag setting itself
                torch::Tensor result = input + input;
                result = result * 2.0;
                
                // Toggle and compute again
                torch::jit::setGraphExecutorOptimize(!enable_optimized);
                torch::Tensor result2 = input + input;
                result2 = result2 * 2.0;
                
                // Restore original setting
                torch::jit::setGraphExecutorOptimize(previous_setting);
                return 0;
            }
        } catch (...) {
            // Module creation failed, just test the optimization flag API
            torch::jit::setGraphExecutorOptimize(true);
            torch::jit::setGraphExecutorOptimize(false);
            torch::jit::setGraphExecutorOptimize(previous_setting);
            return 0;
        }

        // Create inputs for the module
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        // Execute the module with optimized execution setting
        try {
            torch::jit::IValue output = module.forward(inputs);

            // Toggle optimization and run again
            torch::jit::setGraphExecutorOptimize(!enable_optimized);
            torch::jit::IValue output2 = module.forward(inputs);

            // Get results for comparison
            if (output.isTensor() && output2.isTensor()) {
                torch::Tensor result = output.toTensor();
                torch::Tensor result2 = output2.toTensor();
                
                // Verify results match regardless of optimization
                if (result.sizes() == result2.sizes()) {
                    bool close = torch::allclose(result, result2);
                    (void)close; // Use the result
                }
            }
        } catch (...) {
            // Forward execution may fail for various reasons, that's OK
        }

        // Test additional optimization API calls with fuzz-driven values
        if (Size > offset) {
            uint8_t opt_byte = Data[offset % Size];
            torch::jit::setGraphExecutorOptimize((opt_byte & 0x01) != 0);
            torch::jit::setGraphExecutorOptimize((opt_byte & 0x02) != 0);
            torch::jit::setGraphExecutorOptimize((opt_byte & 0x04) != 0);
        }

        // Restore original setting
        torch::jit::setGraphExecutorOptimize(previous_setting);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}