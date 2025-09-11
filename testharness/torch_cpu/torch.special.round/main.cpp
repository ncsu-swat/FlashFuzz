#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.round operation
        torch::Tensor result = torch::special::round(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Variant 1: Apply round with decimals parameter if available
            if (variant % 3 == 0 && offset < Size) {
                int64_t decimals = static_cast<int8_t>(Data[offset++]); // Use int8_t to allow negative values
                torch::Tensor result_with_decimals = torch::round(input, decimals);
            }
            
            // Variant 2: Apply round to a view of the tensor
            if (variant % 3 == 1 && !input.sizes().empty() && input.dim() > 0) {
                torch::Tensor view;
                if (input.numel() > 0) {
                    view = input.view({-1});
                    torch::Tensor result_view = torch::special::round(view);
                }
            }
            
            // Variant 3: Apply round to a slice of the tensor
            if (variant % 3 == 2 && input.dim() > 0 && input.size(0) > 1) {
                torch::Tensor slice = input.slice(0, 0, input.size(0) / 2);
                torch::Tensor result_slice = torch::special::round(slice);
            }
        }
        
        // Try in-place version if available
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Clone the input to avoid modifying the original
            torch::Tensor input_clone = input.clone();
            
            // Apply in-place round if the tensor type supports it
            if (input_clone.is_floating_point() || input_clone.is_complex()) {
                input_clone.round_();
            }
        }
        
        // Try with different output types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType output_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only attempt to change dtype for valid conversions
            try {
                torch::Tensor result_cast = torch::special::round(input).to(output_dtype);
            } catch (const c10::Error &) {
                // Silently catch dtype conversion errors
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
