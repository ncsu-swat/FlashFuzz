#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for tensor creation
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for n (order of Hermite polynomial)
        torch::Tensor n_tensor;
        if (offset < Size) {
            n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create input tensor for x (input values)
        torch::Tensor x_tensor;
        if (offset < Size) {
            x_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for x, use n as x
            x_tensor = n_tensor;
        }
        
        // Try different variants of the function
        
        // Variant 1: Call with both x and n tensors
        torch::Tensor result1 = torch::special::hermite_polynomial_he(x_tensor, n_tensor);
        
        // Variant 2: Call with scalar x and tensor n
        if (x_tensor.numel() > 0) {
            torch::Scalar x_scalar = x_tensor.item();
            torch::Tensor result2 = torch::special::hermite_polynomial_he(x_scalar, n_tensor);
        }
        
        // Variant 3: Call with tensor x and scalar n
        if (n_tensor.numel() > 0) {
            torch::Scalar n_scalar = n_tensor.item();
            torch::Tensor result3 = torch::special::hermite_polynomial_he(x_tensor, n_scalar);
        }
        
        // Edge cases with extreme values
        if (offset + 1 < Size) {
            // Create tensors with extreme values
            uint8_t extreme_selector = Data[offset++];
            
            // Based on the selector, try different extreme cases
            if (extreme_selector % 4 == 0) {
                // Very large n values
                torch::Tensor large_n = torch::ones({2, 2}, torch::kInt64) * 1000000;
                torch::Tensor result_large_n = torch::special::hermite_polynomial_he(x_tensor, large_n);
            } else if (extreme_selector % 4 == 1) {
                // Negative n values
                torch::Tensor neg_n = torch::ones({2, 2}, torch::kInt64) * -1;
                torch::Tensor result_neg_n = torch::special::hermite_polynomial_he(x_tensor, neg_n);
            } else if (extreme_selector % 4 == 2) {
                // Very large x values
                torch::Tensor large_x = torch::ones({2, 2}, torch::kFloat32) * 1e20;
                torch::Tensor result_large_x = torch::special::hermite_polynomial_he(large_x, n_tensor);
            } else {
                // NaN and Inf values in x
                torch::Tensor special_x = torch::ones({3, 3}, torch::kFloat32);
                special_x[0][0] = std::numeric_limits<float>::quiet_NaN();
                special_x[1][1] = std::numeric_limits<float>::infinity();
                special_x[2][2] = -std::numeric_limits<float>::infinity();
                torch::Tensor result_special_x = torch::special::hermite_polynomial_he(special_x, n_tensor);
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