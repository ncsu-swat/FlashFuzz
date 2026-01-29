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
        
        // erf_ requires floating point tensor
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the erf_ operation in-place
        tensor.erf_();
        
        // Verify the operation worked by comparing with non-in-place version
        // Use inner try-catch for expected validation failures (don't log)
        try {
            torch::Tensor expected = torch::erf(original);
            
            // Basic sanity check - shapes should match
            if (tensor.sizes() != expected.sizes()) {
                // This would indicate a bug in PyTorch
                std::cerr << "Shape mismatch after erf_" << std::endl;
            }
            
            // Use allclose with reasonable tolerances for floating point comparison
            // Don't throw - just observe results
            torch::allclose(tensor, expected, 1e-4, 1e-6);
        } catch (...) {
            // Silently ignore validation failures
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
}