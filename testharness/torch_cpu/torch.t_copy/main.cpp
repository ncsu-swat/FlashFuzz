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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::t_copy only works on tensors with <= 2 dimensions
        // For tensors with more dimensions, it will throw an exception
        // which is expected behavior
        
        // Apply torch.t_copy operation
        // t_copy is like t() but always returns a copy, not a view
        torch::Tensor result = torch::t_copy(input_tensor);
        
        // Verify the operation completed by accessing some property
        auto sizes = result.sizes();
        auto input_sizes = input_tensor.sizes();
        
        // Basic sanity checks based on tensor dimension
        // For 2D tensors, dimensions should be swapped
        if (input_tensor.dim() == 2) {
            // Verify transpose: shape [m, n] -> [n, m]
            assert(sizes[0] == input_sizes[1] && sizes[1] == input_sizes[0]);
        }
        // For 1D tensors, t() returns the same 1D tensor (no change)
        else if (input_tensor.dim() == 1) {
            assert(result.dim() == 1);
            assert(sizes[0] == input_sizes[0]);
        }
        // For 0D tensors, result should still be 0D
        else if (input_tensor.dim() == 0) {
            assert(result.dim() == 0);
        }
        
        // Verify it's actually a copy, not a view
        // The result should not share storage with input
        // (though this is implementation detail, just access data to exercise it)
        result.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}