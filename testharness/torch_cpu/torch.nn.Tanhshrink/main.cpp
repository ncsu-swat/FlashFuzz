#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Alternative implementation to verify correctness
        auto tanhshrink_manual = input - torch::tanh(input);
        
        // Ensure the operation doesn't crash with various input types
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            // For floating point types, we can also verify the implementation
            auto diff = torch::abs(tanhshrink - tanhshrink_manual).max().item<double>();
            if (diff > 1e-5) {
                throw std::runtime_error("Tanhshrink implementation mismatch");
            }
        }
        
        // Test with different tensor options
        if (offset + 1 < Size) {
            // Create a new tensor with different options
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply Tanhshrink to the new tensor
            auto tanhshrink2 = torch::nn::functional::tanhshrink(input2);
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
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}