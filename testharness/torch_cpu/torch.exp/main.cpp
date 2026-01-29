#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cstdint>        // For uint64_t

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.exp operation
        torch::Tensor result = torch::exp(input_tensor);
        
        // Try some variants of the operation
        if (offset + 1 < Size) {
            // Use out variant if we have more data
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::exp_out(out_tensor, input_tensor);
            
            // Try in-place variant
            torch::Tensor inplace_tensor = input_tensor.clone();
            inplace_tensor.exp_();
        }
        
        // Try with different options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with non-contiguous tensor (via slice)
            if (option_byte & 0x01) {
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                    // Create non-contiguous tensor via slicing
                    torch::Tensor sliced = input_tensor.slice(0, 0, input_tensor.size(0), 2);
                    torch::Tensor exp_result = torch::exp(sliced);
                }
            }
            
            // Try with transposed tensor for 2D+
            if (option_byte & 0x02) {
                if (input_tensor.dim() >= 2) {
                    torch::Tensor transposed = input_tensor.transpose(0, 1);
                    torch::Tensor exp_result = torch::exp(transposed);
                }
            }
            
            // Try with different dtype - float
            if (option_byte & 0x04) {
                try {
                    torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
                    torch::Tensor float_result = torch::exp(float_tensor);
                } catch (...) {
                    // Silently ignore dtype conversion issues
                }
            }
            
            // Try with different dtype - double
            if (option_byte & 0x08) {
                try {
                    torch::Tensor double_tensor = input_tensor.to(torch::kDouble);
                    torch::Tensor double_result = torch::exp(double_tensor);
                } catch (...) {
                    // Silently ignore dtype conversion issues
                }
            }
            
            // Try with complex dtype
            if (option_byte & 0x10) {
                try {
                    torch::Tensor complex_tensor = input_tensor.to(torch::kComplexFloat);
                    torch::Tensor complex_result = torch::exp(complex_tensor);
                } catch (...) {
                    // Silently ignore dtype conversion issues
                }
            }
            
            // Try with contiguous copy
            if (option_byte & 0x20) {
                torch::Tensor contig = input_tensor.contiguous();
                torch::Tensor exp_result = torch::exp(contig);
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