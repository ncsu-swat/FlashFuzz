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
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.i1 operation (modified Bessel function of the first kind, order 1)
        torch::Tensor result = torch::special::i1(input);
        
        // Force computation by accessing sum (works for any tensor size)
        if (result.defined() && result.numel() > 0) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Try with different input configurations if there's more data
        if (offset + 2 < Size) {
            size_t offset2 = 0;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            torch::Tensor result2 = torch::special::i1(input2);
            
            if (result2.defined() && result2.numel() > 0) {
                volatile float sum2 = result2.sum().item<float>();
                (void)sum2;
            }
        }
        
        // Test with out parameter version if available
        if (input.numel() > 0) {
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::special::i1_out(out_tensor, input);
            
            if (out_tensor.defined() && out_tensor.numel() > 0) {
                volatile float out_sum = out_tensor.sum().item<float>();
                (void)out_sum;
            }
        }
        
        // Test with edge cases - wrap in inner try-catch for expected failures
        try {
            // Test with extreme values
            torch::Tensor extreme_values = torch::tensor({
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN()
            });
            
            torch::Tensor extreme_result = torch::special::i1(extreme_values);
            volatile float extreme_sum = extreme_result.sum().item<float>();
            (void)extreme_sum;
        } catch (...) {
            // Silently ignore expected failures with extreme values
        }
        
        // Test with zero tensor
        if (input.numel() > 0) {
            torch::Tensor zero_tensor = torch::zeros_like(input);
            torch::Tensor zero_result = torch::special::i1(zero_tensor);
            volatile float zero_sum = zero_result.sum().item<float>();
            (void)zero_sum;
        }
        
        // Test with different dtypes if input allows
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor double_result = torch::special::i1(double_input);
            volatile double double_sum = double_result.sum().item<double>();
            (void)double_sum;
        } catch (...) {
            // Silently ignore dtype conversion failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}