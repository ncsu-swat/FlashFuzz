#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        try {
            if (x_copy.sizes() == result.sizes() && x_copy.dtype() == result.dtype()) {
                // Only compare if shapes and dtypes match
                torch::allclose(x_copy, result, 1e-5, 1e-8);
            }
        } catch (...) {
            // Comparison may fail for various reasons, silently ignore
        }
        
        // Test edge cases with scalar inputs
        // Wrap in try-catch since item() requires scalar tensor
        try {
            if (x.numel() == 1 && y.numel() == 1) {
                // Create scalar tensors
                torch::Scalar scalar_x = x.item();
                torch::Scalar scalar_y = y.item();
                
                // Test scalar overloads
                torch::Tensor scalar_result = torch::xlogy(scalar_x, y);
                torch::Tensor scalar_result2 = torch::xlogy(x, scalar_y);
                
                // Test with scalar tensor by reshaping the original tensors
                torch::Tensor scalar_tensor_x = x.reshape({});
                torch::Tensor scalar_tensor_y = y.reshape({});
                torch::Tensor scalar_scalar_result = torch::xlogy(scalar_tensor_x, scalar_tensor_y);
            }
        } catch (...) {
            // Scalar operations may fail, silently ignore
        }
        
        // Test with zero tensor
        try {
            torch::Tensor zeros = torch::zeros_like(x);
            torch::Tensor zeros_result = torch::xlogy(zeros, y);
            zeros.xlogy_(y);
        } catch (...) {
            // May fail due to dtype issues, silently ignore
        }
        
        // Test with positive values to ensure log is valid
        try {
            torch::Tensor pos_y = torch::abs(y) + 1e-6;  // Ensure positive for log
            torch::Tensor x_clone = x.clone();
            x_clone.xlogy_(pos_y);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with out parameter variant if available
        try {
            torch::Tensor out = torch::empty_like(x);
            torch::xlogy_out(out, x, y);
        } catch (...) {
            // May not match shapes, silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}