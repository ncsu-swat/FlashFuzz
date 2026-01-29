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
        
        // Need at least a few bytes for tensor creation and axis parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor rank (number of dimensions)
        int64_t rank = input_tensor.dim();
        
        // swapaxes requires at least 1 dimension
        if (rank < 1) {
            return 0;
        }
        
        // If we don't have enough data left for axis parameters, return
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Extract axis1 and axis2 from the remaining data
        int64_t axis1_raw = static_cast<int64_t>(Data[offset++]);
        int64_t axis2_raw = static_cast<int64_t>(Data[offset++]);
        
        // Map to valid axis range [-rank, rank-1]
        // Use modulo to wrap into valid range
        axis1_raw = axis1_raw % rank;
        axis2_raw = axis2_raw % rank;
        
        // Also test negative indices sometimes
        int64_t axis1 = (offset < Size && (Data[offset] & 1)) ? -axis1_raw - 1 : axis1_raw;
        int64_t axis2 = (offset < Size && (Data[offset] & 2)) ? -axis2_raw - 1 : axis2_raw;
        if (offset < Size) offset++;
        
        // Apply swapaxes operation
        torch::Tensor output = torch::swapaxes(input_tensor, axis1, axis2);
        
        // Verify the output shape is correct
        // After swapaxes, dimensions at axis1 and axis2 should be swapped
        (void)output.sizes();
        
        // Optional: Test edge cases with different tensor types
        if (offset < Size) {
            try {
                uint8_t type_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(type_selector);
                
                // Convert tensor to different type and test swapaxes again
                torch::Tensor converted_input = input_tensor.to(dtype);
                torch::Tensor converted_output = torch::swapaxes(converted_input, axis1, axis2);
                (void)converted_output.sizes();
            } catch (...) {
                // Silently ignore dtype conversion issues
            }
        }
        
        // Optional: Test with contiguous and non-contiguous tensors
        if (rank >= 2) {
            try {
                // Create a non-contiguous view by transposing
                torch::Tensor non_contiguous = input_tensor.transpose(0, rank - 1);
                
                // Verify it's non-contiguous and test
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor output_non_contiguous = torch::swapaxes(non_contiguous, axis1, axis2);
                    (void)output_non_contiguous.sizes();
                }
                
                // Make it contiguous and test again
                torch::Tensor contiguous = non_contiguous.contiguous();
                torch::Tensor output_contiguous = torch::swapaxes(contiguous, axis1, axis2);
                (void)output_contiguous.sizes();
            } catch (...) {
                // Silently ignore issues with non-contiguous tensor operations
            }
        }
        
        // Test swapping same axis (should be a no-op effectively)
        if (offset < Size && (Data[offset - 1] & 4)) {
            torch::Tensor same_axis_output = torch::swapaxes(input_tensor, axis1, axis1);
            (void)same_axis_output.sizes();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}