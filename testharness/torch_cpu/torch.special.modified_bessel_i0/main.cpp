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
        
        // modified_bessel_i0 requires floating-point tensors
        // Convert to float if not already a floating type
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the modified_bessel_i0 operation
        torch::Tensor result = torch::special::modified_bessel_i0(input);
        
        // Force computation by accessing the data
        if (result.defined() && result.numel() > 0) {
            volatile float check = result.sum().item<float>();
            (void)check;
        }
        
        // Try with out variant if we have enough data
        if (offset + 1 < Size) {
            // Create output tensor with same shape and dtype as input
            torch::Tensor out = torch::empty_like(input);
            
            // Apply the operation with out parameter
            torch::special::modified_bessel_i0_out(out, input);
            
            // Force computation
            if (out.defined() && out.numel() > 0) {
                volatile float check = out.sum().item<float>();
                (void)check;
            }
        }
        
        // Test with different dtypes to improve coverage
        if (offset + 2 < Size) {
            try {
                // Test with double precision
                torch::Tensor input_double = input.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::modified_bessel_i0(input_double);
                volatile double check = result_double.sum().item<double>();
                (void)check;
            } catch (...) {
                // Silently ignore dtype conversion issues
            }
        }
        
        // Test with different tensor shapes
        if (Size > 8) {
            try {
                // Create a 2D tensor for additional coverage
                int dim0 = (Data[offset % Size] % 4) + 1;
                int dim1 = (Data[(offset + 1) % Size] % 4) + 1;
                torch::Tensor input_2d = torch::randn({dim0, dim1});
                torch::Tensor result_2d = torch::special::modified_bessel_i0(input_2d);
                volatile float check = result_2d.sum().item<float>();
                (void)check;
            } catch (...) {
                // Silently ignore shape-related issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}