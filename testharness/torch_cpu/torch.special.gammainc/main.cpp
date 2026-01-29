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
        
        // Create input tensors for gammainc(a, x)
        // gammainc requires two input tensors: a and x
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float and ensure valid ranges for gammainc
        // a must be positive, x must be non-negative
        a = a.to(torch::kFloat32).abs() + 0.01f;  // Ensure positive
        x = x.to(torch::kFloat32).abs();          // Ensure non-negative
        
        // Apply the torch.special.gammainc operation
        // gammainc(a, x) computes the regularized lower incomplete gamma function
        torch::Tensor result = torch::special::gammainc(a, x);
        
        // Test with different dtypes
        if (offset + 4 < Size) {
            torch::Tensor a_double = a.to(torch::kFloat64);
            torch::Tensor x_double = x.to(torch::kFloat64);
            torch::Tensor result_double = torch::special::gammainc(a_double, x_double);
        }
        
        // Test scalar inputs if we have enough data
        if (offset + 2 < Size) {
            // Create scalar tensors with valid values
            float a_val = static_cast<float>(Data[offset++]) / 10.0f + 0.1f;  // Positive
            float x_val = static_cast<float>(Data[offset++]) / 10.0f;          // Non-negative
            
            torch::Tensor a_scalar = torch::tensor(a_val);
            torch::Tensor x_scalar = torch::tensor(x_val);
            
            // Test with scalar inputs
            torch::Tensor result_scalar = torch::special::gammainc(a_scalar, x_scalar);
        }
        
        // Test with broadcasting if we have enough data
        if (offset + 4 < Size) {
            // Create tensors with different shapes for broadcasting
            torch::Tensor a_broadcast = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset < Size) {
                torch::Tensor x_broadcast = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure valid ranges
                a_broadcast = a_broadcast.to(torch::kFloat32).abs() + 0.01f;
                x_broadcast = x_broadcast.to(torch::kFloat32).abs();
                
                // Test with broadcasting - inner try-catch for shape mismatches
                try {
                    torch::Tensor result_broadcast = torch::special::gammainc(a_broadcast, x_broadcast);
                } catch (...) {
                    // Silently catch broadcasting failures
                }
            }
        }
        
        // Test with extreme values
        if (offset + 2 < Size) {
            // Small positive a and large x
            torch::Tensor a_small = torch::tensor(0.001f);
            torch::Tensor x_large = torch::tensor(100.0f);
            torch::Tensor result_extreme1 = torch::special::gammainc(a_small, x_large);
            
            // Large a and small x
            torch::Tensor a_large = torch::tensor(100.0f);
            torch::Tensor x_small = torch::tensor(0.001f);
            torch::Tensor result_extreme2 = torch::special::gammainc(a_large, x_small);
            
            // Test with zero x (boundary case)
            torch::Tensor x_zero = torch::tensor(0.0f);
            torch::Tensor result_zero = torch::special::gammainc(a_small, x_zero);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}