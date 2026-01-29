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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for torch.igammac
        // igammac requires two inputs: a and x
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // igammac requires float or double dtype
        if (a.scalar_type() != torch::kFloat && a.scalar_type() != torch::kDouble) {
            a = a.to(torch::kFloat);
        }
        if (x.scalar_type() != torch::kFloat && x.scalar_type() != torch::kDouble) {
            x = x.to(torch::kFloat);
        }
        
        // Apply the torch.igammac operation
        // igammac(a, x) computes the upper regularized incomplete gamma function
        torch::Tensor result = torch::igammac(a, x);
        
        // Test with different data types
        try {
            // Try with double precision
            torch::Tensor a_double = a.to(torch::kDouble);
            torch::Tensor x_double = x.to(torch::kDouble);
            torch::Tensor result_double = torch::igammac(a_double, x_double);
        } catch (...) {
            // Silently ignore type conversion failures
        }
        
        // Test with edge case values - these may cause domain errors which is expected
        try {
            // Test with positive values (igammac requires a > 0, x >= 0)
            torch::Tensor pos_a = torch::abs(a) + 1e-6;
            torch::Tensor pos_x = torch::abs(x);
            torch::Tensor result_pos = torch::igammac(pos_a, pos_x);
        } catch (...) {
            // Silently ignore domain errors
        }
        
        try {
            // Test with very large values
            torch::Tensor large_a = torch::abs(a) * 100.0f + 1.0f;
            torch::Tensor large_x = torch::abs(x) * 100.0f;
            torch::Tensor result_large = torch::igammac(large_a, large_x);
        } catch (...) {
            // Silently ignore overflow/domain errors
        }
        
        try {
            // Test with very small positive values
            torch::Tensor small_a = torch::abs(a) * 1e-5f + 1e-10f;
            torch::Tensor small_x = torch::abs(x) * 1e-5f;
            torch::Tensor result_small = torch::igammac(small_a, small_x);
        } catch (...) {
            // Silently ignore underflow/domain errors
        }
        
        try {
            // Test with scalar tensors if input is small enough
            if (a.numel() == 1 && x.numel() == 1) {
                torch::Tensor scalar_a = torch::tensor(std::abs(a.item<float>()) + 0.1f);
                torch::Tensor scalar_x = torch::tensor(std::abs(x.item<float>()));
                torch::Tensor scalar_result = torch::igammac(scalar_a, scalar_x);
            }
        } catch (...) {
            // Silently ignore scalar extraction failures
        }
        
        // Test with out parameter variant if available
        try {
            torch::Tensor out = torch::empty_like(result);
            torch::igammac_out(out, a, x);
        } catch (...) {
            // Silently ignore if out variant not supported
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}