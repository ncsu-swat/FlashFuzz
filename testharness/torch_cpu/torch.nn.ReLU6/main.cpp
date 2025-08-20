#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Try with inplace version using clamp_
        torch::Tensor input_copy = input.clone();
        input_copy.clamp_(0, 6);
        
        // Try with different configurations
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            
            // Create another ReLU6 module with inplace option
            torch::nn::ReLU6 relu6_config(torch::nn::ReLU6Options().inplace(inplace));
            torch::Tensor output3 = relu6_config(input.clone());
        }
        
        // Try with different tensor types
        if (offset + 1 < Size) {
            // Create a tensor with a different dtype
            size_t new_offset = offset;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            // Apply ReLU6 to the new tensor
            torch::Tensor output4 = relu6(input2);
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        torch::Tensor empty_output = relu6(empty_tensor);
        
        // Try with scalar tensor
        if (offset + 1 < Size) {
            float scalar_value = *reinterpret_cast<const float*>(Data + offset);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor scalar_output = relu6(scalar_tensor);
        }
        
        // Try with extreme values
        torch::Tensor extreme_values = torch::tensor({-1000.0f, -6.0f, -1.0f, 0.0f, 1.0f, 6.0f, 1000.0f});
        torch::Tensor extreme_output = relu6(extreme_values);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}