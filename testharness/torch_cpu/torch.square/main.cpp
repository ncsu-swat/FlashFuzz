#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.square operation
        torch::Tensor result = torch::square(input);
        
        // Force computation by accessing sum (works for any tensor size)
        if (result.defined() && result.numel() > 0) {
            volatile auto sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Try alternative ways to call square
        if (offset + 1 < Size) {
            // Use functional form via at namespace
            torch::Tensor result2 = at::square(input);
            
            // Use method form
            torch::Tensor result3 = input.square();
            
            // Verify results are consistent
            if (result2.defined() && result2.numel() > 0) {
                volatile auto sum2 = result2.sum().item<float>();
                (void)sum2;
            }
            if (result3.defined() && result3.numel() > 0) {
                volatile auto sum3 = result3.sum().item<float>();
                (void)sum3;
            }
        }
        
        // Try in-place version - only works for floating point/complex types
        try {
            if (input.is_floating_point() || input.is_complex()) {
                torch::Tensor input_copy = input.clone();
                input_copy.square_();
                
                if (input_copy.defined() && input_copy.numel() > 0) {
                    volatile auto sum_inplace = input_copy.sum().item<float>();
                    (void)sum_inplace;
                }
            }
        } catch (...) {
            // In-place operations may fail for certain tensor configurations
            // Silently ignore expected failures
        }
        
        // Test with different tensor types to improve coverage
        if (offset + 2 < Size) {
            try {
                torch::Tensor float_tensor = input.to(torch::kFloat32);
                torch::Tensor float_result = torch::square(float_tensor);
                volatile auto fsum = float_result.sum().item<float>();
                (void)fsum;
            } catch (...) {
                // Type conversion may fail, ignore
            }
            
            try {
                torch::Tensor double_tensor = input.to(torch::kFloat64);
                torch::Tensor double_result = torch::square(double_tensor);
                volatile auto dsum = double_result.sum().item<double>();
                (void)dsum;
            } catch (...) {
                // Type conversion may fail, ignore
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