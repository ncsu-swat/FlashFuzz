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
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.abs operation - the main API under test
        torch::Tensor result = torch::abs(input_tensor);
        
        // Try the method version
        torch::Tensor method_result = input_tensor.abs();
        
        // Try out parameter version
        try {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::abs_out(out_tensor, input_tensor);
        } catch (...) {
            // Shape/dtype mismatches are expected, ignore silently
        }
        
        // Try the in-place version
        try {
            torch::Tensor inplace_tensor = input_tensor.clone();
            inplace_tensor.abs_();
        } catch (...) {
            // In-place might fail for certain dtypes, ignore silently
        }
        
        // Create another tensor with different properties if we have more data
        if (offset < Size) {
            try {
                torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor another_result = torch::abs(another_tensor);
                
                // Try abs on a complex tensor if possible
                if (another_tensor.is_floating_point()) {
                    torch::Tensor complex_tensor = torch::complex(another_tensor, another_tensor);
                    torch::Tensor complex_abs = torch::abs(complex_tensor);
                }
            } catch (...) {
                // Expected failures for certain tensor configurations
            }
        }
        
        // Force computation to ensure the operations are not optimized away
        (void)result.numel();
        (void)method_result.numel();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}