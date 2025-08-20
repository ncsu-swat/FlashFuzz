#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure the tensor is square for matrix inversion
        // We don't add defensive checks, but we need to ensure the tensor has at least 2 dimensions
        // to be able to call inverse on it
        if (input_tensor.dim() < 2) {
            // Add dimensions to make it at least 2D
            input_tensor = input_tensor.unsqueeze(0);
            if (input_tensor.dim() < 2) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Make the tensor square by taking the minimum of the last two dimensions
        // and slicing the tensor to that size
        int64_t last_dim = input_tensor.size(-1);
        int64_t second_last_dim = input_tensor.size(-2);
        int64_t min_dim = std::min(last_dim, second_last_dim);
        
        if (min_dim > 0) {
            input_tensor = input_tensor.slice(-1, 0, min_dim).slice(-2, 0, min_dim);
        }
        
        // Convert to float or complex type if needed for numerical stability
        // inverse requires floating point or complex input
        if (input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kUInt8 || 
            input_tensor.dtype() == torch::kInt16 || 
            input_tensor.dtype() == torch::kInt32 || 
            input_tensor.dtype() == torch::kInt64 ||
            input_tensor.dtype() == torch::kBool) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Apply the operation
        torch::Tensor result;
        
        // Try different variants of the operation
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 2;
            
            switch (variant) {
                case 0:
                    // Basic inversion
                    result = torch::inverse(input_tensor);
                    break;
                    
                case 1:
                    // With out parameter
                    {
                        auto options = torch::TensorOptions().dtype(input_tensor.dtype());
                        torch::Tensor out_tensor = torch::empty_like(input_tensor, options);
                        result = torch::inverse_out(out_tensor, input_tensor);
                    }
                    break;
            }
        } else {
            // Default case if we don't have enough data for variant
            result = torch::inverse(input_tensor);
        }
        
        // Verify the result by multiplying with the original matrix
        // A * A^(-1) should be close to the identity matrix
        if (min_dim > 0) {
            torch::Tensor identity_check = torch::matmul(input_tensor, result);
            
            // Use the result to prevent it from being optimized away
            auto sum = identity_check.sum().item<double>();
            if (std::isnan(sum) || std::isinf(sum)) {
                // This is not an error, just a numerical instability
                // We're testing how the API handles these cases
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