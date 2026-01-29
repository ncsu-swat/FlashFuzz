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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.gammaln operation
        torch::Tensor result = torch::special::gammaln(input);
        
        // Try some variants with options
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with out tensor (may fail if dtype doesn't match)
            try {
                torch::Tensor out = torch::empty_like(input);
                torch::special::gammaln_out(out, input);
            } catch (...) {
                // Silently ignore expected failures
            }
            
            // Try with non-standard memory layout if tensor has multiple dimensions
            if (input.dim() > 1) {
                try {
                    torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                    torch::Tensor result_transposed = torch::special::gammaln(transposed);
                } catch (...) {
                    // Silently ignore expected failures
                }
            }
            
            // Try with different dtypes if we have enough data
            if (offset < Size) {
                auto dtype_selector = Data[offset++] % 2;
                try {
                    if (dtype_selector == 0 && input.dtype() != torch::kDouble) {
                        torch::Tensor double_input = input.to(torch::kDouble);
                        torch::Tensor double_result = torch::special::gammaln(double_input);
                    } else if (dtype_selector == 1 && input.dtype() != torch::kFloat) {
                        torch::Tensor float_input = input.to(torch::kFloat);
                        torch::Tensor float_result = torch::special::gammaln(float_input);
                    }
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            }
        }
        
        // Additional coverage: test with contiguous tensor
        try {
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor contiguous_result = torch::special::gammaln(contiguous_input);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with scalar tensor
        if (input.numel() > 0) {
            try {
                torch::Tensor scalar = input.flatten()[0];
                torch::Tensor scalar_result = torch::special::gammaln(scalar);
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}