#include "fuzzer_utils.h"
#include <iostream>
#include <limits>

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply Tanhshrink operation
        // Tanhshrink(x) = x - tanh(x)
        auto tanhshrink = torch::nn::functional::tanhshrink(input);
        
        // Ensure the operation produces valid output
        (void)tanhshrink;
        
        // Test with different tensor options
        if (offset + 1 < Size) {
            // Create a new tensor with different options
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply Tanhshrink to the new tensor
            auto tanhshrink2 = torch::nn::functional::tanhshrink(input2);
            (void)tanhshrink2;
        }
        
        // Test edge cases with special values if we have enough data
        if (offset + 1 < Size) {
            // Create a tensor with special values
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto special_tensor = torch::empty({4}, options);
            
            // Fill with special values: inf, -inf, nan, 0
            special_tensor[0] = std::numeric_limits<float>::infinity();
            special_tensor[1] = -std::numeric_limits<float>::infinity();
            special_tensor[2] = std::numeric_limits<float>::quiet_NaN();
            special_tensor[3] = 0.0f;
            
            // Apply Tanhshrink to special values
            auto special_result = torch::nn::functional::tanhshrink(special_tensor);
            (void)special_result;
        }
        
        // Test with explicitly float tensor for numerical stability
        if (offset + 1 < Size) {
            torch::Tensor float_input = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                auto float_tensor = float_input.to(torch::kFloat);
                auto result = torch::nn::functional::tanhshrink(float_tensor);
                (void)result;
            } catch (...) {
                // Conversion or operation may fail for certain inputs, ignore
            }
        }
        
        // Test with double precision
        if (offset + 1 < Size) {
            torch::Tensor double_input = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                auto double_tensor = double_input.to(torch::kDouble);
                auto result = torch::nn::functional::tanhshrink(double_tensor);
                (void)result;
            } catch (...) {
                // Conversion or operation may fail for certain inputs, ignore
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