#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <limits>

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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply isneginf operation
        torch::Tensor result = torch::isneginf(input);
        
        // Test with floating point tensors to cover more paths
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            // Create a tensor with special values to ensure we test edge cases
            std::vector<double> special_values = {
                -std::numeric_limits<double>::infinity(),  // -inf (should return true)
                std::numeric_limits<double>::infinity(),   // +inf (should return false)
                std::numeric_limits<double>::quiet_NaN(),  // NaN (should return false)
                -0.0,                                      // -0 (should return false)
                0.0,                                       // +0 (should return false)
                -1.0,                                      // negative number (should return false)
                1.0,                                       // positive number (should return false)
                std::numeric_limits<double>::max(),        // max value
                std::numeric_limits<double>::lowest()      // lowest value
            };
            
            auto options = torch::TensorOptions().dtype(input.dtype());
            torch::Tensor special_tensor = torch::tensor(special_values, options);
            
            // Apply isneginf to special values
            torch::Tensor special_result = torch::isneginf(special_tensor);
        }
        
        // Test with different tensor shapes
        if (offset < Size && input.numel() > 0) {
            try {
                // Reshape to different dimensions if possible
                int64_t numel = input.numel();
                torch::Tensor reshaped = input.reshape({numel});
                torch::Tensor result_1d = torch::isneginf(reshaped);
                
                if (numel > 1) {
                    torch::Tensor reshaped_2d = input.reshape({1, numel});
                    torch::Tensor result_2d = torch::isneginf(reshaped_2d);
                }
            } catch (...) {
                // Silently ignore reshape failures
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
            try {
                // Transpose creates a non-contiguous tensor
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor result_transposed = torch::isneginf(transposed);
            } catch (...) {
                // Silently ignore failures
            }
        }
        
        // Test with sliced tensors
        if (input.dim() >= 1 && input.size(0) > 1) {
            try {
                torch::Tensor sliced = input.slice(0, 0, input.size(0) / 2 + 1);
                torch::Tensor result_sliced = torch::isneginf(sliced);
            } catch (...) {
                // Silently ignore failures
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