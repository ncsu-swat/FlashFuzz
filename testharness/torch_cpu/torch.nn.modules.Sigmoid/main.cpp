#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for Sigmoid
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Sigmoid module
        torch::nn::Sigmoid sigmoid_module;
        
        // Apply Sigmoid operation via module
        torch::Tensor output = sigmoid_module->forward(input);
        
        // Apply sigmoid via functional interface
        torch::Tensor output2 = torch::sigmoid(input);
        
        // Try in-place version if possible (requires floating point)
        if (input.is_floating_point()) {
            torch::Tensor input_clone = input.clone();
            input_clone.sigmoid_();
        }
        
        // Test with additional inputs if we have more data
        if (offset < Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply sigmoid with the module
            torch::Tensor another_output = sigmoid_module->forward(another_input);
            
            // Try with gradients if tensor is floating point
            if (another_input.is_floating_point()) {
                torch::Tensor grad_input = another_input.detach().clone().requires_grad_(true);
                torch::Tensor grad_output = sigmoid_module->forward(grad_input);
                
                // Try backward if tensor has non-zero elements
                if (grad_output.numel() > 0) {
                    try {
                        grad_output.sum().backward();
                    } catch (...) {
                        // Silently ignore backward errors (expected for some inputs)
                    }
                }
            }
        }
        
        // Test with different tensor dimensions
        if (offset + 4 < Size) {
            // Create a multi-dimensional tensor
            int dim1 = (Data[offset++] % 4) + 1;  // 1-4
            int dim2 = (Data[offset++] % 4) + 1;  // 1-4
            
            torch::Tensor multi_dim_input = torch::rand({dim1, dim2});
            torch::Tensor multi_dim_output = sigmoid_module->forward(multi_dim_input);
            
            // Verify output shape matches input shape
            (void)multi_dim_output;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}