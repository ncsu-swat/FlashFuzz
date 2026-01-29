#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For INFINITY, NAN

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
        
        // Create input tensor for sinc operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply sinc operation to the input tensor
        torch::Tensor result = torch::sinc(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Create a clone to test in-place operation if available
            torch::Tensor input_clone = input.clone();
            
            // Test sinc_ (in-place) operation on the clone
            try {
                input_clone.sinc_();
            } catch (const std::exception&) {
                // Ignore - in-place may not work for all dtypes
            }
            
            // Test with different input dtype if possible
            if (offset + 2 < Size) {
                uint8_t dtype_selector = Data[offset++];
                auto output_dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                try {
                    torch::Tensor converted_input = input.to(output_dtype);
                    torch::Tensor result3 = torch::sinc(converted_input);
                } catch (const std::exception&) {
                    // Ignore exceptions from dtype conversion or unsupported dtype
                }
            }
        }
        
        // Test edge cases with special values if we have enough data
        if (offset + 4 < Size) {
            try {
                // Create a tensor with special values like 0, inf, -inf, NaN
                // Use from_blob or explicit tensor creation
                float special_values[] = {0.0f, INFINITY, -INFINITY, NAN};
                torch::Tensor special_input = torch::from_blob(
                    special_values, {2, 2}, torch::kFloat).clone();
                torch::Tensor special_result = torch::sinc(special_input);
            } catch (const std::exception&) {
                // Ignore exceptions from special values
            }
        }
        
        // Test with output tensor (out parameter variant)
        if (offset + 1 < Size) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input);
                torch::sinc_out(out_tensor, input);
            } catch (const std::exception&) {
                // Ignore - out variant may not support all cases
            }
        }
        
        // Test with requires_grad for autograd coverage
        if (offset < Size && input.is_floating_point()) {
            try {
                torch::Tensor grad_input = input.detach().clone().set_requires_grad(true);
                torch::Tensor grad_result = torch::sinc(grad_input);
                if (grad_result.numel() > 0) {
                    grad_result.sum().backward();
                }
            } catch (const std::exception&) {
                // Ignore autograd exceptions
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