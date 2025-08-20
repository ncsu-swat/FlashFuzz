#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply erfinv operation
        torch::Tensor result = torch::erfinv(input);
        
        // Try out-of-place version with options
        torch::TensorOptions options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
        torch::Tensor out = torch::empty_like(input, options);
        torch::erfinv_out(out, input);
        
        // Try in-place version if supported for the dtype
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            input_copy.erfinv_();
        }
        
        // Try with values in the valid range for erfinv (-1 to 1)
        if (input.numel() > 0 && input.is_floating_point()) {
            torch::Tensor clamped_input = torch::clamp(input, -0.99, 0.99);
            torch::Tensor clamped_result = torch::erfinv(clamped_input);
        }
        
        // Try with edge cases
        if (offset + 1 < Size) {
            uint8_t edge_case_selector = Data[offset++];
            
            // Create tensors with specific values based on the selector
            if (edge_case_selector % 5 == 0) {
                // Test with tensor of ones (edge of valid domain)
                torch::Tensor ones = torch::ones_like(input);
                torch::Tensor result_ones = torch::erfinv(ones);
            } else if (edge_case_selector % 5 == 1) {
                // Test with tensor of negative ones (other edge of valid domain)
                torch::Tensor neg_ones = -torch::ones_like(input);
                torch::Tensor result_neg_ones = torch::erfinv(neg_ones);
            } else if (edge_case_selector % 5 == 2) {
                // Test with zeros (should be zero)
                torch::Tensor zeros = torch::zeros_like(input);
                torch::Tensor result_zeros = torch::erfinv(zeros);
            } else if (edge_case_selector % 5 == 3) {
                // Test with values outside valid range
                torch::Tensor outside_range = torch::ones_like(input) * 2.0;
                torch::Tensor result_outside = torch::erfinv(outside_range);
            } else {
                // Test with NaN values
                torch::Tensor nan_tensor = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor result_nan = torch::erfinv(nan_tensor);
            }
        }
        
        // Try with different dtypes if we have more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // Create a new tensor with a different dtype
            torch::ScalarType target_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input to the new dtype if possible
            try {
                torch::Tensor converted_input = input.to(target_dtype);
                torch::Tensor converted_result = torch::erfinv(converted_input);
            } catch (const std::exception&) {
                // Some dtype conversions might not be valid, that's fine
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