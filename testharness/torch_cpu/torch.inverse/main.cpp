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
                square_dim = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
                if (square_dim * square_dim > total_elements) {
                    square_dim -= 1;
                }
                if (square_dim < 1) {
                    square_dim = 1;
                }
            }
            
            // Flatten and take only what we need for square matrix
            input_tensor = input_tensor.flatten().slice(0, 0, square_dim * square_dim);
            input_tensor = input_tensor.reshape({square_dim, square_dim});
        }
        
        // Convert to float type if needed for inverse operation
        if (!input_tensor.is_floating_point() && !input_tensor.is_complex()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Try to make the matrix more likely to be invertible by adding a small value to the diagonal
        // This is done by adding epsilon * identity matrix
        if (offset < Size) {
            float epsilon = 0.01f + static_cast<float>(Data[offset % Size]) / 255.0f;
            torch::Tensor identity = torch::eye(input_tensor.size(0), input_tensor.options());
            input_tensor = input_tensor + identity * epsilon;
        }
        
        // Apply the inverse operation
        torch::Tensor result;
        try {
            result = torch::inverse(input_tensor);
        } catch (const std::exception &) {
            // Matrix may be singular, which is expected for some inputs
            return 0;
        }
        
        // Verify the inverse by multiplying with original
        if (result.numel() > 0 && input_tensor.numel() > 0) {
            torch::Tensor identity_check = torch::matmul(input_tensor, result);
            
            // Use the result to prevent optimization from removing the computation
            if (identity_check.numel() > 0) {
                volatile float sum = identity_check.sum().item<float>();
                (void)sum;
            }
        }
        
        // Also test batched inverse if we have enough data
        if (Size > 16 && offset < Size - 4) {
            int64_t batch_size = 1 + (Data[offset % Size] % 3);  // 1-3 batches
            int64_t mat_size = input_tensor.size(0);
            
            // Create batched input
            torch::Tensor batched_input = input_tensor.unsqueeze(0).expand({batch_size, mat_size, mat_size}).clone();
            
            try {
                torch::Tensor batched_result = torch::inverse(batched_input);
                volatile float batch_sum = batched_result.sum().item<float>();
                (void)batch_sum;
            } catch (const std::exception &) {
                // Batched inverse may fail, which is fine
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}