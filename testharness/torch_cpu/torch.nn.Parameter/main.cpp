#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 3) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a Parameter from the tensor
        // Test with different requires_grad values
        bool requires_grad = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Create the Parameter - torch::nn::Parameter is just a tensor with requires_grad
        torch::Tensor parameter = tensor.requires_grad_(requires_grad);
        
        // Test basic Parameter properties
        auto param_data = parameter.data();
        auto param_grad = parameter.grad();
        
        // Test Parameter operations
        if (requires_grad) {
            // Only perform operations that require gradients if requires_grad is true
            
            // Create a simple operation to generate gradients
            torch::Tensor output = parameter.mean();
            
            // Backpropagate
            output.backward();
            
            // Access the gradient
            auto grad_after = parameter.grad();
            
            // Test if gradient was properly computed
            if (grad_after.defined()) {
                // Perform some operation with the gradient
                auto grad_sum = grad_after.sum();
            }
        }
        
        // Test cloning
        auto cloned_param = parameter.clone();
        
        // Test detach
        auto detached = parameter.detach();
        
        // Test to method (changing device/dtype)
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            auto converted = parameter.to(dtype);
        }
        
        // Test Parameter with empty tensor
        if (offset < Size && Data[offset++] % 5 == 0) {
            std::vector<int64_t> empty_shape = {0};
            auto empty_tensor = torch::empty(empty_shape);
            torch::Tensor empty_param = empty_tensor.requires_grad_(requires_grad);
        }
        
        // Test Parameter with scalar tensor
        if (offset < Size && Data[offset++] % 5 == 0) {
            auto scalar_tensor = torch::tensor(3.14);
            torch::Tensor scalar_param = scalar_tensor.requires_grad_(requires_grad);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}