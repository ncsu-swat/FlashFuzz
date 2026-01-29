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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for log_ndtr
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // log_ndtr works best with floating point types
        // Convert to float if not already floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the torch.special.log_ndtr operation
        // log_ndtr computes log of the cumulative distribution function of standard normal
        torch::Tensor result = torch::special::log_ndtr(input);
        
        // Verify result is defined and force computation
        if (result.defined() && result.numel() > 0) {
            // Use sum to force computation without assuming specific shape
            volatile float sum_val = result.sum().item<float>();
            (void)sum_val;
        }
        
        // Try with out variant if we have enough data left
        if (offset + 2 < Size) {
            try {
                // Create output tensor with same shape and dtype as input
                torch::Tensor output = torch::empty_like(input);
                
                // Call the out variant
                torch::special::log_ndtr_out(output, input);
                
                // Force computation
                if (output.defined() && output.numel() > 0) {
                    volatile float out_sum = output.sum().item<float>();
                    (void)out_sum;
                }
            } catch (const std::exception &) {
                // Silently catch expected failures (shape/dtype issues)
            }
        }
        
        // Try with different input configurations
        if (offset + 2 < Size) {
            try {
                torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert to double to test different floating point types
                if (!input2.is_floating_point()) {
                    input2 = input2.to(torch::kFloat64);
                } else {
                    input2 = input2.to(torch::kFloat64);
                }
                
                torch::Tensor result2 = torch::special::log_ndtr(input2);
                
                // Force computation
                if (result2.defined() && result2.numel() > 0) {
                    volatile double sum2 = result2.sum().item<double>();
                    (void)sum2;
                }
            } catch (const std::exception &) {
                // Silently catch expected failures
            }
        }
        
        // Test with specific edge case values if we have data
        if (offset < Size) {
            try {
                // Test with extreme values that exercise log_ndtr edge cases
                uint8_t val = Data[offset++];
                float scale = static_cast<float>(val) / 25.5f - 5.0f; // Range approximately -5 to 5
                
                torch::Tensor edge_input = torch::tensor({scale, -scale, 0.0f, -10.0f, 10.0f});
                torch::Tensor edge_result = torch::special::log_ndtr(edge_input);
                
                volatile float edge_sum = edge_result.sum().item<float>();
                (void)edge_sum;
            } catch (const std::exception &) {
                // Silently catch expected failures
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