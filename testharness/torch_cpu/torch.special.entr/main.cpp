#include "fuzzer_utils.h"
#include <iostream>
#include <limits>

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.entr operation
        // entr(x) = -x * log(x) for x > 0, 0 for x == 0, and -inf for x < 0
        torch::Tensor result = torch::special::entr(input);
        
        // Test edge cases with modified tensors
        uint8_t shift_val = (offset < Size) ? Data[offset++] : 1;
        
        // Create a tensor with some negative values to test edge case
        torch::Tensor neg_input = input - static_cast<float>(shift_val);
        torch::Tensor neg_result = torch::special::entr(neg_input);
        
        // Create a tensor with zeros to test another edge case
        torch::Tensor zero_input = torch::zeros_like(input);
        torch::Tensor zero_result = torch::special::entr(zero_input);
        
        // Create a tensor with very small positive values
        torch::Tensor small_input = input.abs() * 1e-10f + 1e-15f;
        torch::Tensor small_result = torch::special::entr(small_input);
        
        // Create a tensor with very large values
        torch::Tensor large_input = input.abs() * 1e10f + 1.0f;
        torch::Tensor large_result = torch::special::entr(large_input);
        
        // Test with NaN values using torch::where
        torch::Tensor nan_val = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
        torch::Tensor nan_input = torch::where(input > 0, nan_val, input);
        torch::Tensor nan_result = torch::special::entr(nan_input);
        
        // Test with Inf values using torch::where
        torch::Tensor inf_val = torch::full_like(input, std::numeric_limits<float>::infinity());
        torch::Tensor inf_input = torch::where(input > 0, inf_val, input);
        torch::Tensor inf_result = torch::special::entr(inf_input);
        
        // Test with negative infinity
        torch::Tensor neg_inf_val = torch::full_like(input, -std::numeric_limits<float>::infinity());
        torch::Tensor neg_inf_input = torch::where(input < 0, neg_inf_val, input);
        torch::Tensor neg_inf_result = torch::special::entr(neg_inf_input);
        
        // Test with non-contiguous tensor by slicing
        if (input.dim() > 0 && input.size(0) > 1) {
            torch::Tensor strided = input.slice(0, 0, input.size(0), 2);
            torch::Tensor strided_result = torch::special::entr(strided);
        }
        
        // Test with double precision
        torch::Tensor double_input = input.to(torch::kDouble);
        torch::Tensor double_result = torch::special::entr(double_input);
        
        // Test with half precision if available (may throw on CPU)
        try {
            torch::Tensor half_input = input.to(torch::kFloat16);
            torch::Tensor half_result = torch::special::entr(half_input);
        } catch (...) {
            // Half precision may not be supported on CPU
        }
        
        // Test with specific probability-like values (0 to 1 range)
        torch::Tensor prob_input = torch::sigmoid(input);
        torch::Tensor prob_result = torch::special::entr(prob_input);
        
        // Test with ones (should give 0)
        torch::Tensor ones_input = torch::ones_like(input);
        torch::Tensor ones_result = torch::special::entr(ones_input);
        
        // Prevent compiler from optimizing away results
        (void)result;
        (void)neg_result;
        (void)zero_result;
        (void)small_result;
        (void)large_result;
        (void)nan_result;
        (void)inf_result;
        (void)neg_inf_result;
        (void)double_result;
        (void)prob_result;
        (void)ones_result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}