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
        
        // Need at least 2 bytes for the n parameter and the input tensor
        if (Size < 2) {
            return 0;
        }
        
        // Extract n (order of derivative) from the first byte
        // Limit n to reasonable values (0-10) since large n can be slow
        int64_t n = static_cast<int64_t>(Data[offset++]) % 11;
        
        // Create input tensor for polygamma - use floating point types
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor (polygamma requires float/double)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply polygamma operation
        torch::Tensor result = torch::special::polygamma(n, input);
        
        // Optional: Test with different n values to increase coverage
        if (offset < Size) {
            int64_t n2 = static_cast<int64_t>(Data[offset++]) % 11;
            torch::Tensor result2 = torch::special::polygamma(n2, input);
        }
        
        // Test with edge case n values
        if (input.numel() > 0) {
            // Try with n = 0 (digamma function)
            try {
                torch::Tensor result_digamma = torch::special::polygamma(0, input);
            } catch (...) {
                // May fail with certain inputs
            }
            
            // Try with n = 1 (trigamma function)
            try {
                torch::Tensor result_trigamma = torch::special::polygamma(1, input);
            } catch (...) {
                // May fail with certain inputs
            }
            
            // Try with n = 2 (tetragamma)
            try {
                torch::Tensor result_tetra = torch::special::polygamma(2, input);
            } catch (...) {
                // May fail with certain inputs
            }
        }
        
        // Test out parameter variant if available
        if (input.numel() > 0) {
            try {
                torch::Tensor out = torch::empty_like(input);
                torch::special::polygamma_out(out, n, input);
            } catch (...) {
                // May fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}