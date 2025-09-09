#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various dtypes and shapes
        auto tensor_info = generate_tensor_info(Data, Size, offset);
        
        // Create tensor with different data types that are valid for logical operations
        torch::Tensor input;
        
        // Test with boolean tensors (most common case)
        if (tensor_info.dtype == torch::kBool) {
            input = generate_tensor(tensor_info, Data, Size, offset);
        }
        // Test with integer types (should work with logical_not)
        else if (tensor_info.dtype == torch::kInt8 || tensor_info.dtype == torch::kInt16 || 
                 tensor_info.dtype == torch::kInt32 || tensor_info.dtype == torch::kInt64 ||
                 tensor_info.dtype == torch::kUInt8) {
            input = generate_tensor(tensor_info, Data, Size, offset);
        }
        // Test with floating point types
        else if (tensor_info.dtype == torch::kFloat32 || tensor_info.dtype == torch::kFloat64 ||
                 tensor_info.dtype == torch::kFloat16) {
            input = generate_tensor(tensor_info, Data, Size, offset);
        }
        // Default to boolean if unsupported type
        else {
            tensor_info.dtype = torch::kBool;
            input = generate_tensor(tensor_info, Data, Size, offset);
        }

        // Test torch::logical_not with the generated tensor
        torch::Tensor result = torch::logical_not(input);
        
        // Verify result properties
        if (result.numel() != input.numel()) {
            throw std::runtime_error("logical_not changed tensor size");
        }
        
        // Result should always be boolean type
        if (result.dtype() != torch::kBool) {
            throw std::runtime_error("logical_not result is not boolean");
        }
        
        // Test in-place version if input is boolean
        if (input.dtype() == torch::kBool) {
            torch::Tensor input_copy = input.clone();
            torch::logical_not_(input_copy);
            
            // Verify in-place result
            if (!torch::allclose(input_copy, result)) {
                throw std::runtime_error("In-place logical_not produces different result");
            }
        }
        
        // Test with scalar tensors
        if (input.numel() > 0) {
            torch::Tensor scalar_input = input.flatten()[0];
            torch::Tensor scalar_result = torch::logical_not(scalar_input);
            
            if (scalar_result.numel() != 1) {
                throw std::runtime_error("logical_not on scalar should return scalar");
            }
        }
        
        // Test edge cases with empty tensors
        if (offset < Size) {
            torch::Tensor empty_tensor = torch::empty({0}, input.options());
            torch::Tensor empty_result = torch::logical_not(empty_tensor);
            
            if (empty_result.numel() != 0) {
                throw std::runtime_error("logical_not on empty tensor should return empty");
            }
        }
        
        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size - 1) {
            uint8_t device_choice = Data[offset++];
            if (device_choice % 2 == 0) {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::Tensor cuda_result = torch::logical_not(cuda_input);
                
                if (!cuda_result.is_cuda()) {
                    throw std::runtime_error("CUDA logical_not result should be on CUDA");
                }
                
                // Compare with CPU result
                torch::Tensor cpu_result_from_cuda = cuda_result.to(torch::kCPU);
                if (!torch::allclose(result, cpu_result_from_cuda)) {
                    throw std::runtime_error("CUDA and CPU logical_not results differ");
                }
            }
        }
        
        // Test with special values for floating point inputs
        if (input.is_floating_point() && input.numel() > 0) {
            // Test with tensor containing inf, -inf, nan, 0.0, -0.0
            std::vector<double> special_values = {
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0, -0.0, 1.0, -1.0
            };
            
            for (double val : special_values) {
                torch::Tensor special_tensor = torch::full({1}, val, input.options());
                torch::Tensor special_result = torch::logical_not(special_tensor);
                
                // Verify result is boolean
                if (special_result.dtype() != torch::kBool) {
                    throw std::runtime_error("logical_not with special values should return bool");
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}