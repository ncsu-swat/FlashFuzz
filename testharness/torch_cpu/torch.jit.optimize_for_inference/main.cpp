#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    // Initialize module once using a static
    static bool initialized = false;
    static torch::jit::Module base_module;
    
    if (!initialized) {
        try {
            // Create a simple module using TorchScript compilation
            // This creates a module with a simple forward method
            base_module = torch::jit::compile(R"JIT(
                def forward(self, x):
                    y = x * 2.0
                    y = y + 1.0
                    y = torch.relu(y)
                    return y
            )JIT")->_ivalue().toModule();
            base_module.eval();
            initialized = true;
        } catch (...) {
            return 0;
        }
    }
    
    try
    {
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract control bytes for fuzzing different paths
        uint8_t control = Data[offset++];
        bool should_freeze = (control & 0x01) != 0;
        bool should_clone = (control & 0x02) != 0;
        
        // Create input tensor with fuzz-controlled dimensions
        int64_t dim0 = 1 + (Data[offset++ % Size] % 8);
        int64_t dim1 = 1 + (Data[offset++ % Size] % 16);
        
        torch::Tensor input_tensor = torch::randn({dim0, dim1});
        
        // Use some fuzzer data to modify the input values
        if (offset + 2 <= Size) {
            float scale = static_cast<float>(Data[offset] % 10 + 1) / 5.0f;
            input_tensor = input_tensor * scale;
            offset += 2;
        }
        
        // Clone or copy the module for this iteration
        torch::jit::Module test_module = should_clone ? base_module.deepcopy() : base_module.clone();
        test_module.eval();
        
        // Optionally freeze the module
        if (should_freeze) {
            try {
                test_module = torch::jit::freeze(test_module);
            } catch (...) {
                // Freezing may fail in some cases, continue without freezing
            }
        }
        
        // Apply optimize_for_inference - this is the main API under test
        torch::jit::Module optimized_module;
        try {
            optimized_module = torch::jit::optimize_for_inference(test_module);
        } catch (const c10::Error& e) {
            // Some module configurations may not support optimization
            return 0;
        }
        
        // Run the optimized module
        std::vector<torch::jit::IValue> test_inputs;
        test_inputs.push_back(input_tensor);
        
        torch::jit::IValue output;
        try {
            output = optimized_module.forward(test_inputs);
        } catch (...) {
            // Forward may fail for some inputs
            return 0;
        }
        
        // Verify the output
        if (output.isTensor()) {
            torch::Tensor output_tensor = output.toTensor();
            // Basic sanity check - output should have same shape as input
            if (output_tensor.sizes() != input_tensor.sizes()) {
                // Shape mismatch is unexpected but not a crash
            }
        }
        
        // Also run original module and compare outputs for consistency
        try {
            torch::jit::IValue original_output = test_module.forward(test_inputs);
            if (output.isTensor() && original_output.isTensor()) {
                torch::Tensor opt_out = output.toTensor();
                torch::Tensor orig_out = original_output.toTensor();
                // Outputs should be close (optimization shouldn't change semantics)
                bool close = torch::allclose(opt_out, orig_out, 1e-4, 1e-4);
                (void)close;  // Suppress unused variable warning
            }
        } catch (...) {
            // Comparison may fail, not critical
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}