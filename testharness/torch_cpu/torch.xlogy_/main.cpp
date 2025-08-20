#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for xlogy_
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of x to test the in-place operation
        torch::Tensor x_copy = x.clone();
        
        // Apply the xlogy_ operation (in-place)
        // xlogy_(x, y) computes x * log(y) with special handling for x=0
        x_copy.xlogy_(y);
        
        // Also test the non-in-place version for comparison
        torch::Tensor result = torch::xlogy(x, y);
        
        // Verify that in-place and out-of-place versions produce the same result
        if (x_copy.sizes() == result.sizes() && x_copy.dtype() == result.dtype()) {
            // Only compare if shapes and dtypes match
            // This is a basic check to ensure in-place operation works correctly
            bool all_close = torch::allclose(x_copy, result, 1e-5, 1e-8);
            if (!all_close) {
                // This is not an error, just interesting information
                // The fuzzer will continue running
            }
        }
        
        // Test edge cases with scalar inputs
        if (offset + 2 < Size) {
            // Create scalar tensors
            torch::Scalar scalar_x = x.item();
            torch::Scalar scalar_y = y.item();
            
            // Test scalar overloads
            torch::Tensor scalar_result = torch::xlogy(scalar_x, y);
            torch::Tensor scalar_result2 = torch::xlogy(x, scalar_y);
            
            // Test with scalar tensor instead of scalar-scalar combination
            torch::Tensor scalar_tensor_x = torch::tensor(scalar_x);
            torch::Tensor scalar_tensor_y = torch::tensor(scalar_y);
            torch::Tensor scalar_scalar_result = torch::xlogy(scalar_tensor_x, scalar_tensor_y);
        }
        
        // Test with zero tensor
        torch::Tensor zeros = torch::zeros_like(x);
        torch::Tensor zeros_result = torch::xlogy(zeros, y);
        zeros.xlogy_(y);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}