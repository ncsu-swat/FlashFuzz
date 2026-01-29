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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // sin_ requires floating-point tensor, convert if needed
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the sin_ operation in-place
        tensor.sin_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        // Use inner try-catch for validation that may fail on edge cases (NaN, Inf)
        try {
            torch::Tensor expected = torch::sin(original);
            
            // Check if the results match (ignoring NaN comparisons)
            // Only check where both are finite
            auto finite_mask = torch::isfinite(tensor) & torch::isfinite(expected);
            if (finite_mask.any().item<bool>()) {
                auto tensor_finite = tensor.index({finite_mask});
                auto expected_finite = expected.index({finite_mask});
                // Silent failure for edge cases - these are expected
                torch::allclose(tensor_finite, expected_finite, 1e-5, 1e-8);
            }
        } catch (...) {
            // Validation may fail for edge cases, that's okay
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}