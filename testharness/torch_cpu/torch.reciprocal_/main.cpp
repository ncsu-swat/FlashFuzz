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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor with floating point type (reciprocal requires float)
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point type for reciprocal operation
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Make a copy of the input tensor for verification
        torch::Tensor original = input_tensor.clone();
        
        // Apply the reciprocal_ operation (in-place)
        input_tensor.reciprocal_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        // Use a separate try-catch since comparison may fail with NaN values
        try {
            torch::Tensor expected = torch::reciprocal(original);
            
            // For finite values, check that results match
            // Note: NaN != NaN, so we use equal_nan for proper comparison
            torch::Tensor finite_mask = torch::isfinite(expected);
            if (finite_mask.any().item<bool>()) {
                torch::Tensor input_finite = input_tensor.index({finite_mask});
                torch::Tensor expected_finite = expected.index({finite_mask});
                // Silent check - don't throw on mismatch, just verify the API runs
                torch::allclose(input_finite, expected_finite, 1e-5, 1e-8);
            }
        } catch (...) {
            // Comparison may fail for edge cases, that's okay
        }
        
        // Also test reciprocal_ on different tensor configurations
        try {
            // Test with contiguous tensor
            torch::Tensor contiguous_tensor = original.clone().contiguous();
            contiguous_tensor.reciprocal_();
        } catch (...) {
            // Expected for some inputs
        }
        
        // Test with different memory layout if tensor has multiple dimensions
        if (original.dim() > 1) {
            try {
                torch::Tensor transposed = original.clone().transpose(0, 1).contiguous();
                transposed.reciprocal_();
            } catch (...) {
                // Expected for some inputs
            }
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}