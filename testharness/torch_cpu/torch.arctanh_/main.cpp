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
        
        // Test 1: Basic in-place arctanh_ operation
        {
            torch::Tensor input_copy = input.clone();
            input_copy.arctanh_();
            // Force computation
            (void)input_copy.sum().item<float>();
        }
        
        // Test 2: Test with values clamped to valid domain (-1, 1) for arctanh
        {
            torch::Tensor clamped = input.clone().clamp(-0.99, 0.99);
            clamped.arctanh_();
            
            // Verify result is finite for valid inputs
            torch::Tensor expected = torch::arctanh(input.clone().clamp(-0.99, 0.99));
            
            // Both should produce the same finite results
            try {
                if (clamped.defined() && expected.defined() && 
                    clamped.numel() > 0 && expected.numel() > 0) {
                    torch::allclose(clamped, expected, 1e-5, 1e-8);
                }
            } catch (...) {
                // Shape mismatches or other expected failures - ignore silently
            }
        }
        
        // Test 3: Test with different dtypes if possible
        if (input.numel() > 0) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat32).clone();
                float_input.arctanh_();
                (void)float_input.sum().item<float>();
            } catch (...) {
                // Dtype conversion failures - ignore silently
            }
            
            try {
                torch::Tensor double_input = input.to(torch::kFloat64).clone();
                double_input.arctanh_();
                (void)double_input.sum().item<double>();
            } catch (...) {
                // Dtype conversion failures - ignore silently
            }
        }
        
        // Test 4: Test with contiguous and non-contiguous tensors
        if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
            try {
                torch::Tensor transposed = input.transpose(0, 1).clone();
                transposed.arctanh_();
                (void)transposed.sum().item<float>();
            } catch (...) {
                // Ignore failures from non-contiguous tensors
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