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
        
        // Create input tensor for n (order of the polynomial)
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
        
        // Try different variants of the function call
        
        // Variant 1: Call with both n and x tensors
        torch::Tensor result1 = torch::special::laguerre_polynomial_l(x_tensor, n_tensor);
        
        // Variant 2: If n is a scalar-like tensor, try with scalar n
        if (n_tensor.numel() == 1) {
            int64_t n_scalar = n_tensor.item<int64_t>();
            torch::Tensor result2 = torch::special::laguerre_polynomial_l(x_tensor, n_scalar);
        }
        
        // Variant 3: If x is a scalar-like tensor, try with scalar x
        if (x_tensor.numel() == 1) {
            double x_scalar = x_tensor.item<double>();
            torch::Tensor result3 = torch::special::laguerre_polynomial_l(x_scalar, n_tensor);
        }
        
        // Variant 4: Try with negative n values (edge case)
        if (n_tensor.numel() > 0) {
            torch::Tensor neg_n = -n_tensor.abs();
            torch::Tensor result4 = torch::special::laguerre_polynomial_l(x_tensor, neg_n);
        }
        
        // Variant 5: Try with very large n values (edge case)
        if (n_tensor.numel() > 0) {
            torch::Tensor large_n = n_tensor * 1000;
            torch::Tensor result5 = torch::special::laguerre_polynomial_l(x_tensor, large_n);
        }
        
        // Variant 6: Try with extreme x values (edge case)
        if (x_tensor.numel() > 0) {
            torch::Tensor extreme_x = x_tensor * 1e10;
            torch::Tensor result6 = torch::special::laguerre_polynomial_l(extreme_x, n_tensor);
        }
        
        // Variant 7: Try with NaN values in x (edge case)
        if (x_tensor.numel() > 0 && (x_tensor.dtype() == torch::kFloat || x_tensor.dtype() == torch::kDouble)) {
            torch::Tensor nan_x = x_tensor.clone();
            nan_x.index_put_({0}, std::numeric_limits<double>::quiet_NaN());
            torch::Tensor result7 = torch::special::laguerre_polynomial_l(nan_x, n_tensor);
        }
        
        // Variant 8: Try with infinity values in x (edge case)
        if (x_tensor.numel() > 0 && (x_tensor.dtype() == torch::kFloat || x_tensor.dtype() == torch::kDouble)) {
            torch::Tensor inf_x = x_tensor.clone();
            inf_x.index_put_({0}, std::numeric_limits<double>::infinity());
            torch::Tensor result8 = torch::special::laguerre_polynomial_l(inf_x, n_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}