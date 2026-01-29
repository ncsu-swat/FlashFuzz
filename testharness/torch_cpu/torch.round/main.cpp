#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.round operation
        torch::Tensor rounded_tensor = torch::round(input_tensor);
        
        // Try different test variations if we have more data
        if (offset + 1 < Size) {
            uint8_t test_selector = Data[offset++];
            
            if (test_selector % 3 == 0) {
                // Test with out parameter
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                torch::round_out(out_tensor, input_tensor);
            } else if (test_selector % 3 == 1) {
                // Test with different dtypes - convert to float first if needed
                try {
                    torch::Tensor float_tensor = input_tensor.to(torch::kFloat32);
                    torch::Tensor rounded_float = torch::round(float_tensor);
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            } else {
                // Test with double precision
                try {
                    torch::Tensor double_tensor = input_tensor.to(torch::kFloat64);
                    torch::Tensor rounded_double = torch::round(double_tensor);
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            }
        }
        
        // Try inplace version if we have more data
        if (offset < Size) {
            uint8_t inplace_selector = Data[offset++];
            
            if (inplace_selector % 2 == 0) {
                // Create a copy to avoid modifying the original tensor
                // Inplace round requires floating point tensor
                try {
                    torch::Tensor inplace_tensor = input_tensor.to(torch::kFloat32).clone();
                    inplace_tensor.round_();
                } catch (...) {
                    // Silently ignore inplace operation failures
                }
            }
        }
        
        // Test with contiguous vs non-contiguous tensors
        if (offset < Size && input_tensor.dim() >= 2) {
            uint8_t contiguous_selector = Data[offset++];
            if (contiguous_selector % 2 == 0) {
                try {
                    // Create non-contiguous tensor via transpose
                    torch::Tensor transposed = input_tensor.transpose(0, 1);
                    torch::Tensor rounded_transposed = torch::round(transposed);
                } catch (...) {
                    // Silently ignore failures on non-contiguous tensors
                }
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