#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation and axis parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor rank (number of dimensions)
        int64_t rank = input_tensor.dim();
        
        // If we don't have enough data left for axis parameters, return
        if (offset + 2 >= Size) {
            return 0;
        }
        
        // Extract axis1 and axis2 from the remaining data
        int64_t axis1_raw = static_cast<int64_t>(Data[offset++]);
        int64_t axis2_raw = static_cast<int64_t>(Data[offset++]);
        
        // Handle negative indices and ensure they're within bounds
        // We don't do bounds checking here to allow testing of out-of-bounds behavior
        int64_t axis1 = axis1_raw % (2 * rank + 1) - rank;
        int64_t axis2 = axis2_raw % (2 * rank + 1) - rank;
        
        // Apply swapaxes operation
        torch::Tensor output;
        output = torch::swapaxes(input_tensor, axis1, axis2);
        
        // Optional: Test edge cases with different tensor types
        if (offset < Size) {
            uint8_t type_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(type_selector);
            
            // Convert tensor to different type and test swapaxes again
            torch::Tensor converted_input = input_tensor.to(dtype);
            torch::Tensor converted_output = torch::swapaxes(converted_input, axis1, axis2);
        }
        
        // Optional: Test with contiguous and non-contiguous tensors
        if (rank >= 2 && offset < Size) {
            // Create a non-contiguous view by transposing
            torch::Tensor non_contiguous = input_tensor.transpose(0, rank - 1);
            
            // Verify it's non-contiguous
            if (!non_contiguous.is_contiguous()) {
                torch::Tensor output_non_contiguous = torch::swapaxes(non_contiguous, axis1, axis2);
            }
            
            // Make it contiguous and test again
            torch::Tensor contiguous = non_contiguous.contiguous();
            torch::Tensor output_contiguous = torch::swapaxes(contiguous, axis1, axis2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}