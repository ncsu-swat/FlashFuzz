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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // For inverse, we need a square matrix (2D tensor with equal dimensions)
        // If the tensor is not 2D, reshape it to a square matrix if possible
        if (input_tensor.dim() != 2 || input_tensor.size(0) != input_tensor.size(1)) {
            int64_t total_elements = input_tensor.numel();
            
            // Find a suitable square dimension
            int64_t square_dim = 1;
            if (total_elements > 0) {
                square_dim = static_cast<int64_t>(std::sqrt(total_elements));
                if (square_dim * square_dim > total_elements) {
                    square_dim -= 1;
                }
            }
            
            // Reshape to square matrix if possible, otherwise create a small square matrix
            if (square_dim > 0) {
                input_tensor = input_tensor.reshape({square_dim, square_dim});
            } else {
                // Create a small identity matrix as fallback
                input_tensor = torch::eye(2, input_tensor.options());
            }
        }
        
        // Convert to float or complex type if needed for inverse operation
        if (!input_tensor.is_floating_point() && !input_tensor.is_complex()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Try to make the matrix invertible by adding a small value to the diagonal
        // This is just to increase the chances of getting an invertible matrix
        if (offset < Size) {
            float diag_add = static_cast<float>(Data[offset % Size]) / 255.0f;
            torch::Tensor diag = torch::diag(torch::ones_like(input_tensor.diag())) * diag_add;
            input_tensor = input_tensor + diag;
        }
        
        // Apply the inverse operation
        torch::Tensor result;
        result = torch::inverse(input_tensor);
        
        // Optional: verify the inverse by multiplying with original
        if (result.numel() > 0 && input_tensor.numel() > 0) {
            torch::Tensor identity_check = torch::matmul(input_tensor, result);
            
            // Use the result to prevent optimization from removing the computation
            if (identity_check.numel() > 0) {
                volatile float sum = identity_check.sum().item<float>();
                (void)sum;
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
