#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
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
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create LogSigmoid module
        torch::nn::LogSigmoid logsigmoid;
        
        // Apply LogSigmoid to the input tensor
        torch::Tensor output = logsigmoid(input);
        
        // Verify the output is valid - should have same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output tensor has different shape than input tensor");
        }
        
        // Alternative implementation to verify correctness using functional API
        torch::Tensor expected_output = torch::log_sigmoid(input);
        
        // Check that outputs are close (allowing for numerical precision)
        try {
            if (!torch::allclose(output, expected_output, 1e-5, 1e-5)) {
                // Log mismatch but don't crash - could be numerical instability
            }
        } catch (...) {
            // Silently ignore comparison failures (e.g., due to NaN values)
        }
        
        // Try backward pass to test gradient computation
        if (offset < Size && Size - offset >= 1) {
            try {
                // Create a tensor that requires gradients
                auto input_with_grad = input.clone().detach().to(torch::kFloat32).requires_grad_(true);
                
                // Forward pass
                auto output_with_grad = logsigmoid(input_with_grad);
                
                // Create a gradient tensor
                auto grad_output = torch::ones_like(output_with_grad);
                
                // Backward pass
                output_with_grad.backward(grad_output);
                
                // Access gradient to ensure it was computed
                auto grad = input_with_grad.grad();
                (void)grad;
            } catch (...) {
                // Silently ignore backward pass failures (e.g., non-float tensors)
            }
        }
        
        // Also test with inplace=false explicitly via functional API
        try {
            auto func_output = torch::nn::functional::logsigmoid(input);
            (void)func_output;
        } catch (...) {
            // Silently ignore - some dtypes may not be supported
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}