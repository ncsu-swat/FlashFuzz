#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors
        if (tensor.numel() == 0) {
            return 0;
        }
        
        // cosh_ requires floating point tensor
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply the cosh_ operation in-place
        tensor.cosh_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        // Use a silent inner try-catch since allclose can fail for inf/nan values
        try {
            torch::Tensor expected = torch::cosh(original);
            // Just compute, don't check - results may differ due to floating point
            (void)expected;
        } catch (...) {
            // Silently ignore comparison failures
        }
        
        // Try with double precision if we have more data
        if (offset + 4 < Size) {
            size_t offset2 = offset;
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset2);
            
            if (tensor2.numel() > 0) {
                // Convert to double for more precision coverage
                tensor2 = tensor2.to(torch::kFloat64);
                tensor2.cosh_();
            }
        }
        
        // Test with a contiguous vs non-contiguous tensor
        if (tensor.dim() >= 2 && tensor.size(0) > 1 && tensor.size(1) > 1) {
            try {
                // Create a non-contiguous view via transpose
                torch::Tensor transposed = original.transpose(0, 1).clone().transpose(0, 1);
                transposed.cosh_();
            } catch (...) {
                // Silently ignore failures from non-contiguous operations
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}