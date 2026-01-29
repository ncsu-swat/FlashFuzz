#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for x (input values) - should be floating point
        torch::Tensor x_tensor;
        if (offset < Size) {
            x_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure x is floating point for hermite polynomial evaluation
            if (!x_tensor.is_floating_point()) {
                x_tensor = x_tensor.to(torch::kFloat32);
            }
        } else {
            return 0;
        }
        
        // Create input tensor for n (order of Hermite polynomial) - should be integer
        torch::Tensor n_tensor;
        if (offset < Size) {
            n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to integer type and take absolute value (order must be non-negative)
            n_tensor = n_tensor.abs().to(torch::kInt64);
            // Clamp to reasonable range to avoid numerical issues
            n_tensor = torch::clamp(n_tensor, 0, 20);
        } else {
            // If we don't have enough data for n, use a simple integer tensor
            n_tensor = torch::tensor({0, 1, 2, 3}, torch::kInt64);
        }
        
        // Variant 1: Call with x tensor and n tensor (both need to be broadcastable)
        try {
            torch::Tensor result1 = torch::special::hermite_polynomial_h(x_tensor, n_tensor);
            (void)result1;
        } catch (const std::exception &) {
            // Shape mismatch or other expected errors - silently ignore
        }
        
        // Variant 2: Call with x tensor and n scalar
        try {
            // Use a fixed small integer for n_scalar
            int64_t n_val = static_cast<int64_t>(Data[0] % 21); // 0-20 range
            torch::Tensor result2 = torch::special::hermite_polynomial_h(x_tensor, n_val);
            (void)result2;
        } catch (const std::exception &) {
            // Expected errors - silently ignore
        }
        
        // Variant 3: Call with x scalar and n tensor
        try {
            if (x_tensor.numel() == 1) {
                double x_val = x_tensor.item<double>();
                torch::Tensor result3 = torch::special::hermite_polynomial_h(x_val, n_tensor);
                (void)result3;
            }
        } catch (const std::exception &) {
            // Expected errors - silently ignore
        }
        
        // Variant 4: Try with out variant
        try {
            // Create output tensor with appropriate shape
            auto out_sizes = x_tensor.sizes().vec();
            torch::Tensor out_tensor = torch::empty(out_sizes, x_tensor.options());
            
            // Need n_tensor to have compatible shape
            torch::Tensor n_scalar_tensor = torch::tensor({static_cast<int64_t>(Data[0] % 21)}, torch::kInt64);
            
            torch::special::hermite_polynomial_h_out(out_tensor, x_tensor, n_scalar_tensor);
            (void)out_tensor;
        } catch (const std::exception &) {
            // Expected errors - silently ignore
        }
        
        // Variant 5: Test with different dtypes
        try {
            torch::Tensor x_double = x_tensor.to(torch::kFloat64);
            int64_t n_val = static_cast<int64_t>(Data[0] % 11);
            torch::Tensor result5 = torch::special::hermite_polynomial_h(x_double, n_val);
            (void)result5;
        } catch (const std::exception &) {
            // Expected errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}