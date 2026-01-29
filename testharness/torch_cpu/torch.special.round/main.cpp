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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch::round operation (not torch::special::round which doesn't exist)
        torch::Tensor result = torch::round(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Variant 1: Apply round with decimals parameter
            if (variant % 3 == 0 && offset < Size) {
                int64_t decimals = static_cast<int8_t>(Data[offset++]); // Use int8_t to allow negative values
                // Clamp decimals to reasonable range to avoid potential issues
                decimals = std::max(static_cast<int64_t>(-10), std::min(decimals, static_cast<int64_t>(10)));
                try {
                    torch::Tensor result_with_decimals = torch::round(input, decimals);
                } catch (const c10::Error &) {
                    // Silently catch errors - decimals parameter may not be supported for all dtypes
                }
            }
            
            // Variant 2: Apply round to a view of the tensor
            if (variant % 3 == 1 && !input.sizes().empty() && input.dim() > 0) {
                try {
                    if (input.numel() > 0) {
                        torch::Tensor view = input.view({-1});
                        torch::Tensor result_view = torch::round(view);
                    }
                } catch (const c10::Error &) {
                    // Silently catch view errors
                }
            }
            
            // Variant 3: Apply round to a slice of the tensor
            if (variant % 3 == 2 && input.dim() > 0 && input.size(0) > 1) {
                try {
                    torch::Tensor slice = input.slice(0, 0, input.size(0) / 2);
                    torch::Tensor result_slice = torch::round(slice);
                } catch (const c10::Error &) {
                    // Silently catch slice errors
                }
            }
        }
        
        // Try in-place version
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Clone the input to avoid modifying the original
            torch::Tensor input_clone = input.clone();
            
            // Apply in-place round if the tensor type supports it
            if (input_clone.is_floating_point()) {
                try {
                    input_clone.round_();
                } catch (const c10::Error &) {
                    // Silently catch in-place errors
                }
            }
        }
        
        // Try with different output types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType output_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only attempt to change dtype for valid conversions
            try {
                torch::Tensor result_cast = torch::round(input).to(output_dtype);
            } catch (const c10::Error &) {
                // Silently catch dtype conversion errors
            }
        }
        
        // Additional coverage: test with contiguous and non-contiguous tensors
        if (offset < Size && input.dim() >= 2) {
            try {
                // Create a non-contiguous tensor via transpose
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                torch::Tensor result_transposed = torch::round(transposed);
            } catch (const c10::Error &) {
                // Silently catch errors
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