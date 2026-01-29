#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Create input tensors for gammaincc(a, x)
        // gammaincc computes the regularized upper incomplete gamma function
        // Requires: a > 0 and x >= 0
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for the special function (it requires floating point)
        a = a.to(torch::kFloat32);
        x = x.to(torch::kFloat32);
        
        // Apply the torch.special.gammaincc operation
        torch::Tensor result = torch::special::gammaincc(a, x);
        
        // Test with absolute values to ensure valid domain (a > 0, x >= 0)
        try {
            torch::Tensor a_valid = torch::abs(a) + 0.001f;  // Ensure a > 0
            torch::Tensor x_valid = torch::abs(x);           // Ensure x >= 0
            torch::Tensor result_valid = torch::special::gammaincc(a_valid, x_valid);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test scalar inputs (only if tensors have exactly one element)
        try {
            if (a.numel() == 1 && x.numel() == 1) {
                // Extract scalar values as float
                float a_val = a.item<float>();
                float x_val = x.item<float>();
                
                // Create tensors from float values
                torch::Tensor a_from_scalar = torch::tensor(a_val);
                torch::Tensor x_from_scalar = torch::tensor(x_val);
                
                torch::Tensor result_scalar1 = torch::special::gammaincc(a_from_scalar, x);
                torch::Tensor result_scalar2 = torch::special::gammaincc(a, x_from_scalar);
                torch::Tensor result_scalar3 = torch::special::gammaincc(a_from_scalar, x_from_scalar);
            }
        } catch (...) {
            // Silently ignore expected failures from scalar operations
        }
        
        // Test broadcasting with different shapes
        if (offset + 4 < Size) {
            try {
                torch::Tensor a_broadcast = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat32);
                if (offset < Size) {
                    torch::Tensor x_broadcast = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat32);
                    torch::Tensor result_broadcast = torch::special::gammaincc(a_broadcast, x_broadcast);
                }
            } catch (...) {
                // Silently ignore expected broadcasting failures
            }
        }
        
        // Test with output tensor
        try {
            torch::Tensor out = torch::empty_like(a);
            torch::special::gammaincc_out(out, a, x);
        } catch (...) {
            // Silently ignore failures (shape mismatch, etc.)
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}