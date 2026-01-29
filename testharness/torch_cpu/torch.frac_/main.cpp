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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // frac_ only works on floating point tensors
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Make a copy of the input tensor to verify the in-place operation
        torch::Tensor original = input_tensor.clone();
        
        // Apply the frac_ operation (in-place)
        // frac_() returns the fractional portion of each element
        input_tensor.frac_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        // For each element x, frac(x) = x - trunc(x)
        // Note: Use trunc instead of floor for correct behavior with negative numbers
        torch::Tensor expected = original - original.trunc();
        
        // Only check correctness for finite values to avoid NaN comparison issues
        if (input_tensor.numel() > 0) {
            try {
                auto finite_mask = torch::isfinite(original);
                if (finite_mask.any().item<bool>()) {
                    auto result_finite = input_tensor.masked_select(finite_mask);
                    auto expected_finite = expected.masked_select(finite_mask);
                    if (result_finite.numel() > 0 && 
                        !torch::allclose(result_finite, expected_finite, 1e-5, 1e-8)) {
                        // Log but don't throw - this helps identify potential issues
                        std::cerr << "Warning: frac_ result differs from expected" << std::endl;
                    }
                }
            } catch (...) {
                // Silently ignore verification errors
            }
        }
        
        // Try with different tensor types to improve coverage
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (!another_tensor.is_floating_point()) {
                another_tensor = another_tensor.to(torch::kFloat64);
            }
            another_tensor.frac_();
        }
        
        // Test with a contiguous vs non-contiguous tensor
        if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
            try {
                torch::Tensor transposed = original.transpose(0, 1).clone();
                transposed.frac_();
            } catch (...) {
                // Silently ignore - some configurations may not work
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