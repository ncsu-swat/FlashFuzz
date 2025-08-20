#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for n (degree of Legendre polynomial)
        torch::Tensor n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor for x (input values)
        if (offset < Size) {
            torch::Tensor x_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try different variants of the function call
            if (Size % 3 == 0) {
                // Variant 1: Call with both n and x tensors
                torch::Tensor result = torch::special::legendre_polynomial_p(x_tensor, n_tensor);
            } else if (Size % 3 == 1) {
                // Variant 2: Call with scalar n and tensor x
                // Extract a scalar from n_tensor if possible
                if (n_tensor.numel() > 0) {
                    torch::Scalar n_scalar = n_tensor.item();
                    torch::Tensor result = torch::special::legendre_polynomial_p(x_tensor, n_scalar);
                }
            } else {
                // Variant 3: Call with tensor n and scalar x
                // Extract a scalar from x_tensor if possible
                if (x_tensor.numel() > 0) {
                    torch::Scalar x_scalar = x_tensor.item();
                    torch::Tensor result = torch::special::legendre_polynomial_p(x_scalar, n_tensor);
                }
            }
        }
        
        // Try with scalar inputs if we have enough data
        if (offset + 1 < Size) {
            // Extract two bytes to use as scalars
            int64_t n_val = static_cast<int64_t>(Data[offset++]);
            double x_val = static_cast<double>(Data[offset++]) / 255.0; // Normalize to [0,1]
            
            // Call with scalar inputs - create tensors from scalars
            torch::Tensor x_tensor = torch::tensor(x_val);
            torch::Tensor n_tensor = torch::tensor(n_val);
            torch::Tensor result = torch::special::legendre_polynomial_p(x_tensor, n_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}