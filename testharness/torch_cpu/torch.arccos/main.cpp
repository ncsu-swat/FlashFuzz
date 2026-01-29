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
        
        // Apply arccos operation
        torch::Tensor result = torch::arccos(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.arccos_();
        }
        
        // Try with different options if we have more data
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should try with different options
            uint8_t option_byte = Data[offset++];
            
            // Try with out tensor
            torch::Tensor out = torch::empty_like(input);
            torch::arccos_out(out, input);
            
            // Try with non-contiguous tensors
            if (input.dim() > 0 && input.size(0) > 1) {
                torch::Tensor non_contiguous = input.transpose(0, input.dim()-1);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor result_non_contiguous = torch::arccos(non_contiguous);
                }
            }
            
            // Try with different dtypes if possible
            if (option_byte % 2 == 0) {
                try {
                    torch::Tensor float_input = input.to(torch::kFloat);
                    torch::Tensor float_result = torch::arccos(float_input);
                } catch (const std::exception&) {
                    // Conversion might fail, that's okay
                }
            }
            
            if (option_byte % 3 == 0) {
                try {
                    torch::Tensor double_input = input.to(torch::kDouble);
                    torch::Tensor double_result = torch::arccos(double_input);
                } catch (const std::exception&) {
                    // Conversion might fail, that's okay
                }
            }
            
            // Try with values clamped to valid range for arccos [-1, 1]
            try {
                torch::Tensor clamped_input = torch::clamp(input.to(torch::kFloat), -1.0, 1.0);
                torch::Tensor clamped_result = torch::arccos(clamped_input);
            } catch (const std::exception&) {
                // Conversion might fail, that's okay
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}