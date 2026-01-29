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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // modified_bessel_i1 requires floating point input
        // Convert to float if not already floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the modified_bessel_i1 operation
        torch::Tensor result = torch::special::modified_bessel_i1(input);
        
        // Access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            // Use sum() instead of item() to handle multi-element tensors
            volatile float sink = result.sum().item<float>();
            (void)sink;
        }
        
        // Try with out variant if we have enough data
        if (offset + 1 < Size) {
            // Create output tensor with same shape and dtype as input
            torch::Tensor out = torch::empty_like(input);
            
            // Apply the operation with out parameter
            torch::special::modified_bessel_i1_out(out, input);
            
            // Access the result
            if (out.defined() && out.numel() > 0) {
                volatile float sink = out.sum().item<float>();
                (void)sink;
            }
        }
        
        // Test with different dtypes based on fuzzer data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset % Size];
            torch::Tensor typed_input;
            
            try {
                if (dtype_selector % 2 == 0) {
                    typed_input = input.to(torch::kFloat64);
                } else {
                    typed_input = input.to(torch::kFloat32);
                }
                
                torch::Tensor typed_result = torch::special::modified_bessel_i1(typed_input);
                if (typed_result.defined() && typed_result.numel() > 0) {
                    volatile double sink = typed_result.sum().item<double>();
                    (void)sink;
                }
            } catch (...) {
                // Silently ignore dtype conversion issues
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