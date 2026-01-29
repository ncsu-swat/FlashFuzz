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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Mish requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create Mish module and apply
        torch::nn::Mish mish_module;
        torch::Tensor output = mish_module->forward(input);
        
        // Use functional version as well
        torch::Tensor output_functional = torch::mish(input);
        
        // Test with requires_grad
        if (offset < Size && (Data[offset++] & 1)) {
            torch::Tensor input_grad = input.clone().detach().requires_grad_(true);
            torch::Tensor out_grad = torch::mish(input_grad);
            // Trigger backward pass
            try {
                out_grad.sum().backward();
            } catch (...) {
                // Backward might fail for certain configurations
            }
        }
        
        // Try with different tensor types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only floating point types are valid for Mish
            if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                try {
                    torch::Tensor input_converted = input.to(dtype);
                    torch::Tensor output_converted = torch::mish(input_converted);
                } catch (...) {
                    // Some dtypes might not be supported on all devices
                }
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat32);
            torch::Tensor empty_output = mish_module->forward(empty_tensor);
        } catch (...) {
            // Empty tensor might not be supported
        }
        
        // Try with scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(3.14f);
            torch::Tensor scalar_output = mish_module->forward(scalar_tensor);
        } catch (...) {
            // Handle exception if scalar input is not supported
        }
        
        // Try with multi-dimensional tensor
        if (offset + 4 <= Size) {
            int dim1 = (Data[offset++] % 8) + 1;
            int dim2 = (Data[offset++] % 8) + 1;
            int dim3 = (Data[offset++] % 8) + 1;
            int dim4 = (Data[offset++] % 8) + 1;
            
            try {
                torch::Tensor multi_dim = torch::randn({dim1, dim2, dim3, dim4});
                torch::Tensor multi_output = torch::mish(multi_dim);
            } catch (...) {
                // Handle potential shape issues
            }
        }
        
        // Test functional mish with inplace option using MishOptions
        try {
            torch::Tensor input_copy = input.clone();
            torch::nn::functional::MishFuncOptions options;
            options.inplace(true);
            torch::nn::functional::mish(input_copy, options);
        } catch (...) {
            // Inplace might not be available or fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}