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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the input tensor for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply atanh_ operation (in-place)
        // Note: atanh is defined for values in (-1, 1), values outside will produce NaN/Inf
        input_copy.atanh_();
        
        // Verify with non-in-place version
        torch::Tensor expected = torch::atanh(input);
        
        // Compare results - use inner try-catch since allclose can fail with NaN values
        if (input.numel() > 0) {
            try {
                // Only compare if we have finite values
                torch::Tensor finite_mask = torch::isfinite(input_copy) & torch::isfinite(expected);
                if (finite_mask.any().item<bool>()) {
                    torch::Tensor copy_masked = input_copy.masked_select(finite_mask);
                    torch::Tensor expected_masked = expected.masked_select(finite_mask);
                    if (copy_masked.numel() > 0) {
                        bool is_close = torch::allclose(copy_masked, expected_masked, 1e-5, 1e-8);
                        if (!is_close) {
                            fuzzer_utils::saveDiffInput(Data, Size, fuzzer_utils::sanitizedTimestamp());
                        }
                    }
                }
            } catch (...) {
                // Ignore comparison failures (e.g., due to NaN handling differences)
            }
        }
        
        // Also test with different tensor types to improve coverage
        if (Size > 4) {
            try {
                // Test with float tensor
                torch::Tensor float_input = input.to(torch::kFloat32).clone();
                float_input.atanh_();
            } catch (...) {
                // Silently ignore type conversion failures
            }
            
            try {
                // Test with double tensor
                torch::Tensor double_input = input.to(torch::kFloat64).clone();
                double_input.atanh_();
            } catch (...) {
                // Silently ignore type conversion failures
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}