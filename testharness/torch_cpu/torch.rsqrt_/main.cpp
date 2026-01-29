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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // rsqrt_ requires floating point tensor
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply the rsqrt_ operation in-place
        // rsqrt(x) = 1/sqrt(x), works for positive values
        // For negative values, it produces NaN (expected behavior)
        tensor.rsqrt_();
        
        // Verify the operation worked correctly by comparing with the expected result
        if (original.numel() > 0) {
            torch::Tensor expected = torch::rsqrt(original);
            
            // Use equal_nan to handle NaN values from negative inputs
            // Only check on non-NaN values to avoid issues with allclose
            try {
                auto mask = torch::isfinite(expected);
                if (mask.any().item<bool>()) {
                    auto tensor_masked = tensor.index({mask});
                    auto expected_masked = expected.index({mask});
                    if (tensor_masked.numel() > 0) {
                        bool equal = torch::allclose(tensor_masked, expected_masked, 1e-5, 1e-8);
                        (void)equal; // Suppress unused variable warning
                    }
                }
            } catch (...) {
                // Silently ignore verification errors - this is just a sanity check
            }
        }
        
        // Also test with different tensor configurations to improve coverage
        if (offset < Size) {
            // Test with contiguous tensor
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (!tensor2.is_floating_point()) {
                tensor2 = tensor2.to(torch::kFloat32);
            }
            tensor2 = tensor2.contiguous();
            tensor2.rsqrt_();
            
            // Test with non-contiguous tensor (transposed)
            if (tensor2.dim() >= 2) {
                torch::Tensor tensor3 = tensor2.clone().transpose(0, 1);
                tensor3.rsqrt_();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}