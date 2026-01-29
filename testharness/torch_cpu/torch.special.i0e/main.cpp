#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Create input tensor for torch.special.i0e
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for special functions (i0e expects floating point)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the torch.special.i0e operation
        // i0e computes the exponentially scaled modified Bessel function of the first kind
        torch::Tensor result = torch::special::i0e(input);
        
        // Verify result is computed
        if (result.defined() && result.numel() > 0) {
            // Force evaluation by accessing sum instead of item (works for any size)
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Try with different input configurations if we have more data
        if (Size - offset > 2) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat64);  // Try double precision
            }
            
            torch::Tensor result2 = torch::special::i0e(input2);
            
            if (result2.defined() && result2.numel() > 0) {
                volatile double sum2 = result2.sum().item<double>();
                (void)sum2;
            }
        }
        
        // Test with edge cases if we have enough data
        if (Size - offset > 2) {
            torch::Tensor edge_input = fuzzer_utils::createTensor(Data, Size, offset);
            if (!edge_input.is_floating_point()) {
                edge_input = edge_input.to(torch::kFloat32);
            }
            
            // Test with large values (i0e is numerically stable for large inputs)
            try {
                torch::Tensor scaled_input = edge_input * 1e10f;
                torch::Tensor result3 = torch::special::i0e(scaled_input);
                
                if (result3.defined() && result3.numel() > 0) {
                    volatile float sum3 = result3.sum().item<float>();
                    (void)sum3;
                }
            } catch (...) {
                // Silently ignore expected failures from extreme values
            }
            
            // Test with negative values (i0e is symmetric: i0e(-x) = i0e(x))
            try {
                torch::Tensor neg_input = -edge_input.abs();
                torch::Tensor result4 = torch::special::i0e(neg_input);
                
                if (result4.defined() && result4.numel() > 0) {
                    volatile float sum4 = result4.sum().item<float>();
                    (void)sum4;
                }
            } catch (...) {
                // Silently ignore expected failures
            }
            
            // Test with very small values near zero
            try {
                torch::Tensor small_input = edge_input * 1e-10f;
                torch::Tensor result5 = torch::special::i0e(small_input);
                
                if (result5.defined() && result5.numel() > 0) {
                    volatile float sum5 = result5.sum().item<float>();
                    (void)sum5;
                }
            } catch (...) {
                // Silently ignore expected failures
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