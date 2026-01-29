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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure the tensor is at least 2D for matrix inversion
        if (input_tensor.dim() < 2) {
            input_tensor = input_tensor.unsqueeze(0);
            if (input_tensor.dim() < 2) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Make the tensor square by taking the minimum of the last two dimensions
        int64_t last_dim = input_tensor.size(-1);
        int64_t second_last_dim = input_tensor.size(-2);
        int64_t min_dim = std::min(last_dim, second_last_dim);
        
        if (min_dim == 0) {
            return 0; // Can't invert empty matrix
        }
        
        input_tensor = input_tensor.slice(-1, 0, min_dim).slice(-2, 0, min_dim);
        
        // Convert to float or complex type - inverse requires floating point or complex input
        if (!input_tensor.is_floating_point() && !input_tensor.is_complex()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Ensure contiguous for better coverage
        input_tensor = input_tensor.contiguous();
        
        torch::Tensor result;
        
        // Try different variants of the operation
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 2;
            
            try {
                switch (variant) {
                    case 0:
                        // torch::inverse - the C++ API for matrix inversion
                        result = torch::inverse(input_tensor);
                        break;
                        
                    case 1:
                        // torch::inverse with out parameter using inverse_out
                        {
                            torch::Tensor out_tensor = torch::empty_like(input_tensor);
                            torch::inverse_out(out_tensor, input_tensor);
                            result = out_tensor;
                        }
                        break;
                }
            } catch (const c10::Error &e) {
                // Expected errors like singular matrix - catch silently
                return 0;
            }
        } else {
            try {
                result = torch::inverse(input_tensor);
            } catch (const c10::Error &e) {
                // Expected errors like singular matrix - catch silently
                return 0;
            }
        }
        
        // Verify the result by multiplying with the original matrix
        // A * A^(-1) should be close to the identity matrix
        try {
            torch::Tensor identity_check = torch::matmul(input_tensor, result);
            
            // Use the result to prevent it from being optimized away
            volatile auto sum = identity_check.sum().item<double>();
            (void)sum;
        } catch (...) {
            // Verification step is not critical - catch silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}