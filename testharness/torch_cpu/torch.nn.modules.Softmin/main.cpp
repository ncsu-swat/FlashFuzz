#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get number of dimensions for valid dim parameter
        int64_t ndim = input.dim();
        if (ndim == 0) {
            // Scalar tensor - softmin requires at least 1D
            return 0;
        }
        
        // Parse dim parameter from the remaining data and constrain to valid range
        int64_t dim = -1;
        if (offset < Size) {
            // Use a single byte to select dimension
            dim = static_cast<int64_t>(Data[offset] % ndim);
            // Also allow negative indexing
            if (offset + 1 < Size && (Data[offset + 1] & 1)) {
                dim = dim - ndim;  // Convert to negative index
            }
            offset += 2;
        }
        
        // Create Softmin module and apply to input
        // Use brace initialization to avoid most vexing parse
        torch::nn::Softmin softmin_module{torch::nn::SoftminOptions(dim)};
        torch::Tensor output = softmin_module->forward(input);
        
        // Try with different dimensions (inner try-catch for expected failures)
        if (offset < Size) {
            try {
                int64_t dim2 = static_cast<int64_t>(Data[offset] % ndim);
                offset++;
                
                torch::nn::Softmin softmin_module2{torch::nn::SoftminOptions(dim2)};
                torch::Tensor output2 = softmin_module2->forward(input);
            } catch (...) {
                // Silently ignore dimension-related failures
            }
        }
        
        // Try with default dimension (-1, last dimension)
        try {
            torch::nn::Softmin default_softmin{torch::nn::SoftminOptions(-1)};
            torch::Tensor default_output = default_softmin->forward(input);
        } catch (...) {
            // Silently ignore if -1 dimension fails
        }
        
        // Try with first dimension (0)
        try {
            torch::nn::Softmin first_dim_softmin{torch::nn::SoftminOptions(0)};
            torch::Tensor first_output = first_dim_softmin->forward(input);
        } catch (...) {
            // Silently ignore failures
        }
        
        // Test with different dtypes if possible
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::nn::Softmin float_softmin{torch::nn::SoftminOptions(dim)};
            torch::Tensor float_output = float_softmin->forward(float_input);
        } catch (...) {
            // Silently ignore dtype conversion failures
        }
        
        // Test with double precision
        try {
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::nn::Softmin double_softmin{torch::nn::SoftminOptions(dim)};
            torch::Tensor double_output = double_softmin->forward(double_input);
        } catch (...) {
            // Silently ignore dtype conversion failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}