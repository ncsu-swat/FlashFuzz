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
        
        // Create input tensors for torch.igammac
        // igammac requires two inputs: a and x
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure tensors have compatible shapes for broadcasting
        // No need to check - let PyTorch handle any shape incompatibilities
        
        // Apply the torch.igammac operation
        // igammac(a, x) computes the complementary incomplete gamma function
        torch::Tensor result = torch::igammac(a, x);
        
        // Optional: test edge cases by creating scalar tensors
        if (Size > offset + 2) {
            // Create scalar tensors for a and x
            torch::Tensor scalar_a = torch::tensor(a.item<float>());
            torch::Tensor scalar_x = torch::tensor(x.item<float>());
            
            // Test igammac with scalar inputs
            torch::Tensor scalar_result = torch::igammac(scalar_a, scalar_x);
        }
        
        // Test with different data types if possible
        if (Size > offset + 4) {
            // Try with double precision
            torch::Tensor a_double = a.to(torch::kDouble);
            torch::Tensor x_double = x.to(torch::kDouble);
            torch::Tensor result_double = torch::igammac(a_double, x_double);
            
            // Try with half precision if supported
            if (torch::cuda::is_available()) {
                torch::Tensor a_half = a.to(torch::kHalf);
                torch::Tensor x_half = x.to(torch::kHalf);
                torch::Tensor result_half = torch::igammac(a_half, x_half);
            }
        }
        
        // Test with extreme values
        if (Size > offset + 2) {
            // Test with very large values
            torch::Tensor large_a = torch::ones_like(a) * 1e10;
            torch::Tensor large_x = torch::ones_like(x) * 1e10;
            torch::Tensor result_large = torch::igammac(large_a, large_x);
            
            // Test with very small positive values
            torch::Tensor small_a = torch::ones_like(a) * 1e-10;
            torch::Tensor small_x = torch::ones_like(x) * 1e-10;
            torch::Tensor result_small = torch::igammac(small_a, small_x);
            
            // Test with zero values (may cause domain errors)
            torch::Tensor zero_a = torch::zeros_like(a);
            torch::Tensor zero_x = torch::zeros_like(x);
            torch::Tensor result_zero = torch::igammac(zero_a, x);
            
            // Test with negative values (may cause domain errors)
            torch::Tensor neg_a = -a;
            torch::Tensor neg_x = -x;
            torch::Tensor result_neg = torch::igammac(neg_a, x);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
