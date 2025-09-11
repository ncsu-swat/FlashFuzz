#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a parameter from the tensor
        // Test with different requires_grad values
        bool requires_grad = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Create a parameter using the tensor
        torch::Tensor parameter = tensor.set_requires_grad(requires_grad);
        
        // Test basic properties of the parameter
        auto param_data = parameter.data();
        auto param_grad = parameter.grad();
        
        // Test parameter operations
        if (requires_grad) {
            // Only perform operations that require gradients if requires_grad is true
            
            // Create a simple operation to generate gradients
            torch::Tensor output = parameter.mean();
            
            // Backpropagate
            output.backward();
            
            // Access the gradient
            auto grad_after = parameter.grad();
        }
        
        // Test cloning the parameter
        auto cloned_param = parameter.clone();
        
        // Test detaching the parameter
        auto detached = parameter.detach();
        
        // Test to_string
        std::string param_str = parameter.toString();
        
        // Test is_leaf property
        bool is_leaf = parameter.is_leaf();
        
        // Test setting requires_grad
        parameter.set_requires_grad(!requires_grad);
        
        // Test other parameter properties
        bool is_contiguous = parameter.is_contiguous();
        auto dtype = parameter.dtype();
        auto device = parameter.device();
        
        // If we have more data, try creating another parameter with different options
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            bool another_requires_grad = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
            torch::Tensor another_param = another_tensor.set_requires_grad(another_requires_grad);
            
            // Test parameter equality
            bool params_equal = parameter.equal(another_param);
            
            // Test parameter addition if shapes match
            try {
                auto sum = parameter + another_param;
            } catch (const std::exception&) {
                // Shapes might not match, that's fine
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
