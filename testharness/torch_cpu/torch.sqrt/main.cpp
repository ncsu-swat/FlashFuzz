#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply sqrt operation
        torch::Tensor result = torch::sqrt(input);
        
        // Try inplace version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.sqrt_();
        }
        
        // Try with out parameter if there's more data
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::sqrt_out(out, input);
        }
        
        // Try with complex tensors specifically if there's more data
        if (offset < Size) {
            // Create a complex tensor if the original wasn't complex
            if (input.is_complex()) {
                torch::Tensor complex_result = torch::sqrt(input);
            } else {
                // Try to create a complex tensor
                try {
                    torch::Tensor complex_input;
                    if (input.dim() > 0) {
                        complex_input = torch::complex(input, input);
                    } else {
                        // For scalar tensors, create a new complex tensor
                        complex_input = torch::complex(torch::tensor(1.0), torch::tensor(1.0));
                    }
                    torch::Tensor complex_result = torch::sqrt(complex_input);
                } catch (const std::exception&) {
                    // Ignore exceptions from complex tensor creation
                }
            }
        }
        
        // Try with negative values to test behavior
        if (offset < Size) {
            try {
                torch::Tensor neg_input;
                if (input.is_floating_point() || input.is_complex()) {
                    neg_input = -torch::abs(input);
                    torch::Tensor neg_result = torch::sqrt(neg_input);
                }
            } catch (const std::exception&) {
                // Ignore exceptions from negative inputs
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