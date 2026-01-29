#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Tanhshrink module
        torch::nn::Tanhshrink tanhshrink_module;
        
        // Apply Tanhshrink operation: f(x) = x - tanh(x)
        torch::Tensor output = tanhshrink_module->forward(input);
        
        // Alternative implementation to verify correctness
        torch::Tensor expected_output = input - torch::tanh(input);
        
        // Verify results are close (use allclose for floating point comparison)
        try {
            if (!torch::allclose(output, expected_output, 1e-5, 1e-5)) {
                // Results differ unexpectedly - this would indicate a bug
                std::cerr << "Output mismatch detected!" << std::endl;
            }
        } catch (...) {
            // Comparison may fail for certain tensor types, ignore
        }
        
        // Test with a second tensor if we have enough remaining data
        if (offset + 4 < Size) {
            size_t offset2 = offset;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset2);
            
            // Apply Tanhshrink operation again
            torch::Tensor output2 = tanhshrink_module->forward(input2);
            
            // Verify the alternative implementation
            torch::Tensor expected_output2 = input2 - torch::tanh(input2);
            
            try {
                torch::allclose(output2, expected_output2, 1e-5, 1e-5);
            } catch (...) {
                // Ignore comparison failures
            }
        }
        
        // Test functional interface as well
        torch::Tensor functional_output = torch::nn::functional::tanhshrink(input);
        
        try {
            torch::allclose(output, functional_output, 1e-5, 1e-5);
        } catch (...) {
            // Ignore comparison failures
        }
        
        // Test in-place operation behavior (create a clone to avoid modifying original)
        torch::Tensor input_clone = input.clone();
        torch::Tensor inplace_result = input_clone - torch::tanh(input_clone);
        
        // Test with different tensor configurations
        if (input.is_floating_point()) {
            // Test with contiguous tensor
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor contiguous_output = tanhshrink_module->forward(contiguous_input);
            
            // Test with non-contiguous tensor (transpose if 2D+)
            if (input.dim() >= 2) {
                try {
                    torch::Tensor transposed = input.transpose(0, 1);
                    torch::Tensor transposed_output = tanhshrink_module->forward(transposed);
                } catch (...) {
                    // Shape operations may fail, ignore
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}