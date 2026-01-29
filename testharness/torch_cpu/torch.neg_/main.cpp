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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the neg_ operation (in-place negation)
        tensor.neg_();
        
        // Verify the operation worked correctly by comparing with manual negation
        // Use equal_nan to handle NaN values properly
        torch::Tensor expected_result = -original;
        
        // Only check for non-NaN values to avoid false positives
        // NaN != NaN by definition, so we need special handling
        try {
            torch::Tensor mask = ~torch::isnan(original);
            if (mask.any().item<bool>()) {
                torch::Tensor tensor_masked = tensor.index({mask});
                torch::Tensor expected_masked = expected_result.index({mask});
                if (!torch::allclose(tensor_masked, expected_masked)) {
                    throw std::runtime_error("neg_ operation produced unexpected results");
                }
            }
        } catch (const c10::Error&) {
            // Silently ignore indexing errors for complex tensor shapes
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}