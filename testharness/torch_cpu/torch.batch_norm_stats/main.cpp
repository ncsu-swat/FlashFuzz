#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // batch_norm_stats requires a floating-point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure the input tensor has at least 2 dimensions for batch norm
        // First dimension is batch size, second is features/channels
        if (input.dim() < 2) {
            // Reshape to at least 2D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1]
                new_shape = {1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, size]
                new_shape = {1, input.size(0)};
            }
            
            if (!new_shape.empty()) {
                input = input.reshape(new_shape);
            }
        }
        
        // Ensure tensor is contiguous (batch_norm_stats may require this)
        input = input.contiguous();
        
        // Get a value for epsilon from the input data
        double epsilon = 1e-5; // Default value
        if (offset + sizeof(float) <= Size) {
            float eps_val;
            std::memcpy(&eps_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is positive and reasonable
            if (std::isfinite(eps_val) && eps_val > 1e-10 && eps_val < 1.0) {
                epsilon = static_cast<double>(eps_val);
            }
        }
        
        // Call batch_norm_stats - returns tuple of (mean, invstd)
        auto result = torch::batch_norm_stats(input, epsilon);
        
        // Unpack the result (mean and inverse standard deviation)
        auto mean = std::get<0>(result);
        auto invstd = std::get<1>(result);
        
        // Perform some operation with the results to ensure they're used
        // Use inner try-catch for expected numerical edge cases
        try {
            auto sum = mean.sum() + invstd.sum();
            (void)sum; // Suppress unused variable warning
        } catch (...) {
            // Numerical issues with sum are acceptable
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}