#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Create ReLU6 module
        torch::nn::ReLU6 relu6;
        
        // Apply ReLU6 to the input tensor
        torch::Tensor output = relu6(input);
        
        // Alternative way to apply ReLU6 using functional API
        torch::Tensor output2 = torch::nn::functional::relu6(input);
        
        // Try with inplace version using clamp_ (silent catch for dtype issues)
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.clamp_(0, 6);
        } catch (...) {
            // Silently ignore - some dtypes may not support inplace clamp
        }
        
        // Try with different configurations
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            
            // Create another ReLU6 module with inplace option
            torch::nn::ReLU6 relu6_config(torch::nn::ReLU6Options().inplace(inplace));
            torch::Tensor output3 = relu6_config(input.clone());
        }
        
        // Try with different tensor types
        if (offset + 1 < Size) {
            size_t new_offset = offset;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            // Apply ReLU6 to the new tensor
            torch::Tensor output4 = relu6(input2);
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_output = relu6(empty_tensor);
        } catch (...) {
            // Silently ignore - empty tensor edge case
        }
        
        // Try with scalar tensor (safe float reading)
        if (offset + sizeof(float) <= Size) {
            float scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN/Inf cases
            if (std::isfinite(scalar_value)) {
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor scalar_output = relu6(scalar_tensor);
            }
        }
        
        // Try with extreme values
        torch::Tensor extreme_values = torch::tensor({-1000.0f, -6.0f, -1.0f, 0.0f, 1.0f, 6.0f, 1000.0f});
        torch::Tensor extreme_output = relu6(extreme_values);
        
        // Try functional API with inplace option
        if (offset < Size) {
            bool func_inplace = Data[offset++] % 2 == 0;
            torch::Tensor func_input = input.clone();
            torch::Tensor func_output = torch::nn::functional::relu6(
                func_input, 
                torch::nn::functional::ReLU6FuncOptions().inplace(func_inplace)
            );
        }
        
        // Test with different dtypes
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor double_output = relu6(double_input);
        } catch (...) {
            // Silently ignore dtype conversion issues
        }
        
        try {
            torch::Tensor half_input = input.to(torch::kHalf);
            torch::Tensor half_output = relu6(half_input);
        } catch (...) {
            // Silently ignore - half precision may not be supported
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}