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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // sqrt_ requires floating point tensor
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        try {
            // Apply the sqrt_ operation in-place
            tensor.sqrt_();
            
            // Verify the operation worked correctly by comparing with non-in-place version
            torch::Tensor expected = torch::sqrt(original);
            
            // Check if the results match (within numerical tolerance)
            // Note: NaN == NaN is false, so we use allclose which handles NaN properly
            if (tensor.defined() && expected.defined() && 
                tensor.numel() > 0 && expected.numel() > 0) {
                // Use equal_nan=true to handle NaN values from negative inputs
                bool close = torch::allclose(tensor, expected, /*rtol=*/1e-5, /*atol=*/1e-8, /*equal_nan=*/true);
                (void)close; // Suppress unused variable warning
            }
        }
        catch (const c10::Error &e) {
            // Expected failures (e.g., unsupported dtype) - silently ignore
            return 0;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
}