#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.sgn operation
        torch::Tensor result = torch::sgn(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Use out variant if we have more data
            try {
                torch::Tensor out = torch::empty_like(input);
                torch::sgn_out(out, input);
            } catch (...) {
                // Silently ignore - out tensor may have incompatible dtype
            }
            
            // Try in-place variant
            // sgn_() works on floating point and complex types
            try {
                if (input.is_floating_point() || input.is_complex()) {
                    torch::Tensor input_copy = input.clone();
                    input_copy.sgn_();
                }
            } catch (...) {
                // Silently ignore in-place operation failures
            }
        }
        
        // Try with different tensor configurations if we have more data
        if (offset + 2 < Size) {
            uint8_t option_byte = Data[offset++];
            
            try {
                // Create a view if possible
                if (option_byte % 3 == 0 && input.numel() > 0) {
                    torch::Tensor view = input.view({-1});
                    torch::sgn(view);
                }
                
                // Try with non-contiguous tensor
                if (option_byte % 3 == 1 && input.dim() > 1 && input.size(0) > 1) {
                    torch::Tensor non_contig = input.transpose(0, input.dim() - 1);
                    if (!non_contig.is_contiguous()) {
                        torch::sgn(non_contig);
                    }
                }
                
                // Try with strided tensor
                if (option_byte % 3 == 2 && input.dim() > 0 && input.size(0) > 1) {
                    torch::Tensor strided = input.slice(0, 0, input.size(0), 2);
                    torch::sgn(strided);
                }
            } catch (...) {
                // Silently ignore view/stride operation failures
            }
        }
        
        // Additional coverage: test with specific tensor types
        if (offset + 1 < Size) {
            uint8_t type_byte = Data[offset++];
            
            try {
                // Test with complex tensor
                if (type_byte % 4 == 0 && input.numel() > 0) {
                    torch::Tensor complex_input = torch::complex(input.to(torch::kFloat), input.to(torch::kFloat));
                    torch::sgn(complex_input);
                }
                
                // Test with zero tensor
                if (type_byte % 4 == 1) {
                    torch::Tensor zeros = torch::zeros_like(input);
                    torch::sgn(zeros);
                }
                
                // Test with negative values
                if (type_byte % 4 == 2 && input.is_floating_point()) {
                    torch::Tensor neg_input = input * -1;
                    torch::sgn(neg_input);
                }
                
                // Test with mixed positive/negative
                if (type_byte % 4 == 3 && input.is_floating_point() && input.numel() > 0) {
                    torch::Tensor mixed = input - input.mean();
                    torch::sgn(mixed);
                }
            } catch (...) {
                // Silently ignore type conversion failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}