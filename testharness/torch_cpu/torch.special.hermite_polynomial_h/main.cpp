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
        
        // Need at least some data to create tensors
        if (Size < 4) {
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
        
        // Try different variants of the function call
        
        // Variant 1: Call with both n and x tensors
        torch::Tensor result1 = torch::special::hermite_polynomial_h(x_tensor, n_tensor);
        
        // Variant 2: Call with x tensor and n scalar
        if (n_tensor.numel() > 0) {
            torch::Scalar n_scalar = n_tensor.item();
            torch::Tensor result2 = torch::special::hermite_polynomial_h(x_tensor, n_scalar);
        }
        
        // Variant 3: Call with x scalar and n tensor
        if (x_tensor.numel() > 0) {
            torch::Scalar x_scalar = x_tensor.item();
            torch::Tensor result3 = torch::special::hermite_polynomial_h(x_scalar, n_tensor);
        }
        
        // Try with out variant if we have enough data
        if (offset < Size) {
            // Create output tensor
            torch::Tensor out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Call with out tensor
            torch::special::hermite_polynomial_h_out(out_tensor, x_tensor, n_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
