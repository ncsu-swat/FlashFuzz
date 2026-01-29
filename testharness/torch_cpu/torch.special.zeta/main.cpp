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
        
        // Create input tensors for torch.special.zeta
        // zeta(x, q) computes the Hurwitz zeta function
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for zeta function (requires floating point)
        if (!x.is_floating_point()) {
            x = x.to(torch::kFloat32);
        }
        
        // Check if we have enough data left for the second tensor
        if (offset < Size) {
            torch::Tensor q = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to float for zeta function
            if (!q.is_floating_point()) {
                q = q.to(torch::kFloat32);
            }
            
            // Apply torch.special.zeta operation
            torch::Tensor result = torch::special::zeta(x, q);
            
            // Test with output tensor
            try {
                torch::Tensor out = torch::empty_like(result);
                torch::special::zeta_out(out, x, q);
            } catch (...) {
                // Shape/broadcast issues - ignore
            }
            
            // Try the scalar version with single-element tensors
            try {
                if (x.numel() == 1 && q.numel() > 0) {
                    auto scalar_x = x.item<double>();
                    torch::Tensor result_scalar_x = torch::special::zeta(scalar_x, q);
                }
            } catch (...) {
                // Scalar extraction or computation issues - ignore
            }
            
            try {
                if (q.numel() == 1 && x.numel() > 0) {
                    auto scalar_q = q.item<double>();
                    torch::Tensor result_scalar_q = torch::special::zeta(x, scalar_q);
                }
            } catch (...) {
                // Scalar extraction or computation issues - ignore
            }
        } else {
            // If we don't have enough data for a second tensor, use default value of 1
            torch::Tensor ones = torch::ones_like(x);
            torch::Tensor result = torch::special::zeta(x, ones);
            
            // Test with output tensor
            try {
                torch::Tensor out = torch::empty_like(result);
                torch::special::zeta_out(out, x, ones);
            } catch (...) {
                // Shape issues - ignore
            }
            
            // Try the scalar version with single-element tensor
            try {
                if (x.numel() == 1) {
                    auto scalar_x = x.item<double>();
                    torch::Tensor result_scalar = torch::special::zeta(scalar_x, ones);
                }
            } catch (...) {
                // Scalar extraction issues - ignore
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