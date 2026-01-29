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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.special.erf
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.erf operation
        torch::Tensor result = torch::special::erf(input);
        
        // Try some edge cases with different tensor views
        if (input.dim() > 0 && input.numel() > 0) {
            try {
                // Test with a slice of the tensor if possible
                torch::Tensor slice = input.slice(0, 0, input.size(0) / 2 + 1);
                torch::Tensor slice_result = torch::special::erf(slice);
            } catch (...) {
                // Silently ignore shape-related errors
            }
            
            try {
                // Test with a transposed tensor if rank >= 2
                if (input.dim() >= 2) {
                    torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                    torch::Tensor transposed_result = torch::special::erf(transposed);
                }
            } catch (...) {
                // Silently ignore
            }
            
            try {
                // Test with a contiguous version if not already contiguous
                if (!input.is_contiguous()) {
                    torch::Tensor contiguous = input.contiguous();
                    torch::Tensor contiguous_result = torch::special::erf(contiguous);
                }
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test with different output types if possible
        try {
            if (input.scalar_type() == torch::kFloat || 
                input.scalar_type() == torch::kDouble) {
                // Test with output to a different dtype
                torch::Tensor result_double = torch::special::erf(input.to(torch::kDouble));
                torch::Tensor result_float = torch::special::erf(input.to(torch::kFloat));
            }
        } catch (...) {
            // Silently ignore dtype conversion issues
        }
        
        // Test with inplace version if available
        try {
            if (input.scalar_type() == torch::kFloat || 
                input.scalar_type() == torch::kDouble ||
                input.scalar_type() == torch::kHalf ||
                input.scalar_type() == torch::kBFloat16) {
                torch::Tensor input_copy = input.clone();
                input_copy.erf_();
            }
        } catch (...) {
            // Silently ignore inplace operation failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}