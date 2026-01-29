#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to apply softmax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // If tensor has dimensions, make sure dim is within valid range
        if (input.dim() > 0) {
            // Allow negative dimensions (PyTorch handles them by wrapping)
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
        } else {
            dim = 0;
        }
        
        // Create Softmax module with the dimension using brace initialization
        // to avoid the "most vexing parse" issue
        torch::nn::Softmax softmax{torch::nn::SoftmaxOptions(dim)};
        
        // Apply softmax to the input tensor
        // Use inner try-catch for expected failures
        try {
            torch::Tensor output = softmax->forward(input);
        } catch (const std::exception &) {
            // Shape/dimension errors are expected
        }
        
        // Try functional softmax
        try {
            torch::Tensor output2 = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(dim));
        } catch (const std::exception &) {
            // Expected failures
        }
        
        // Try with different dimensions if tensor has dimensions
        if (input.dim() > 1) {
            int64_t alt_dim = (dim + 1) % input.dim();
            torch::nn::Softmax alt_softmax{torch::nn::SoftmaxOptions(alt_dim)};
            try {
                torch::Tensor alt_output = alt_softmax->forward(input);
            } catch (const std::exception &) {
                // Expected failures
            }
        }
        
        // Try with last dimension (common use case)
        if (input.dim() > 0) {
            try {
                torch::Tensor default_output = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(-1));
            } catch (const std::exception &) {
                // Expected failures
            }
        }
        
        // Try with first dimension
        if (input.dim() > 0) {
            try {
                torch::Tensor first_dim_output = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(0));
            } catch (const std::exception &) {
                // Expected failures
            }
        }
        
        // Test with specific dtype
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor dtype_output = torch::nn::functional::softmax(
                float_input, 
                torch::nn::functional::SoftmaxFuncOptions(dim).dtype(torch::kFloat32)
            );
        } catch (const std::exception &) {
            // Expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}