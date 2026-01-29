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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Identity module with default configuration
        torch::nn::Identity identity_default;
        
        // Apply Identity operation
        torch::Tensor output_tensor = identity_default->forward(input_tensor);
        
        // Verify output is same as input (identity property)
        // This exercises the comparison code path
        if (!torch::equal(input_tensor, output_tensor)) {
            std::cerr << "Identity property violated!" << std::endl;
        }
        
        // Try with sequential module - use brace initialization to avoid vexing parse
        torch::nn::Sequential sequential{
            torch::nn::Identity()
        };
        torch::Tensor output_sequential = sequential->forward(input_tensor);
        
        // Test chained identity operations - use brace initialization
        torch::nn::Sequential chained{
            torch::nn::Identity(),
            torch::nn::Identity(),
            torch::nn::Identity()
        };
        torch::Tensor output_chained = chained->forward(input_tensor);
        
        // Test with different tensor types if we have enough data
        if (offset + 4 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor another_output = identity_default->forward(another_tensor);
        }
        
        // Test forward with requires_grad tensors
        try {
            torch::Tensor grad_tensor = input_tensor.clone().set_requires_grad(true);
            torch::Tensor grad_output = identity_default->forward(grad_tensor);
        } catch (...) {
            // Silently ignore if gradient operations fail (e.g., for integer tensors)
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}