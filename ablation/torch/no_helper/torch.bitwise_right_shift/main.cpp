#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor configuration parameters
        auto tensor_config = extract_tensor_config(Data, Size, offset);
        if (offset >= Size) return 0;

        // Create input tensor with integral dtype
        std::vector<torch::ScalarType> integral_types = {
            torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64,
            torch::kUInt8
        };
        
        torch::ScalarType input_dtype = integral_types[tensor_config.dtype_index % integral_types.size()];
        
        // Create input tensor
        torch::Tensor input = create_tensor_from_config(tensor_config, input_dtype);
        
        // Extract second tensor/scalar configuration
        if (offset >= Size) return 0;
        auto other_config = extract_tensor_config(Data, Size, offset);
        if (offset >= Size) return 0;

        // Decide whether to use tensor or scalar for 'other'
        bool use_scalar = (Data[offset % Size] % 2) == 0;
        offset++;

        if (use_scalar) {
            // Use scalar for shift amount
            int64_t shift_amount = static_cast<int64_t>(Data[offset % Size]) % 32; // Limit to reasonable range
            offset++;
            
            // Test both positive and negative shifts (though negative is undefined behavior)
            if ((Data[offset % Size] % 4) == 0 && shift_amount > 0) {
                shift_amount = -shift_amount; // Test negative shift (undefined behavior)
            }
            offset++;

            // Test with scalar
            torch::Tensor result1 = torch::bitwise_right_shift(input, shift_amount);
            
            // Test with out parameter
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::bitwise_right_shift_out(out_tensor, input, shift_amount);
            
        } else {
            // Use tensor for shift amounts
            torch::ScalarType other_dtype = integral_types[other_config.dtype_index % integral_types.size()];
            torch::Tensor other = create_tensor_from_config(other_config, other_dtype);
            
            // Clamp shift values to reasonable range to avoid undefined behavior most of the time
            other = torch::clamp(other, 0, 31);
            
            // Occasionally test edge cases with potentially undefined behavior
            if ((Data[offset % Size] % 8) == 0) {
                // Test with large shift values (undefined behavior)
                other = other + 64;
            } else if ((Data[offset % Size] % 8) == 1) {
                // Test with negative shift values (undefined behavior)
                other = -torch::abs(other);
            }
            offset++;

            // Test basic operation
            torch::Tensor result1 = torch::bitwise_right_shift(input, other);
            
            // Test with broadcasting - create tensors with different but compatible shapes
            if ((Data[offset % Size] % 3) == 0 && input.numel() > 1) {
                // Create a smaller tensor that can broadcast
                std::vector<int64_t> broadcast_shape = {1};
                torch::Tensor broadcast_other = torch::randint(0, 8, broadcast_shape, other_dtype);
                torch::Tensor result2 = torch::bitwise_right_shift(input, broadcast_other);
            }
            offset++;

            // Test with out parameter
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::bitwise_right_shift_out(out_tensor, input, other);
            
            // Test type promotion by using different integral types
            if ((Data[offset % Size] % 4) == 0) {
                torch::ScalarType promoted_dtype = integral_types[(other_config.dtype_index + 1) % integral_types.size()];
                torch::Tensor other_promoted = other.to(promoted_dtype);
                torch::Tensor result3 = torch::bitwise_right_shift(input, other_promoted);
            }
            offset++;
        }

        // Test edge cases with specific values
        if ((Data[offset % Size] % 5) == 0) {
            // Test with zero shift
            torch::Tensor zero_shift = torch::zeros({1}, input_dtype);
            torch::Tensor result_zero = torch::bitwise_right_shift(input, zero_shift);
            
            // Test with maximum safe shift for the data type
            int max_shift = 7; // Safe for int8
            if (input_dtype == torch::kInt16) max_shift = 15;
            else if (input_dtype == torch::kInt32) max_shift = 31;
            else if (input_dtype == torch::kInt64) max_shift = 63;
            
            torch::Tensor max_shift_tensor = torch::full({1}, max_shift, input_dtype);
            torch::Tensor result_max = torch::bitwise_right_shift(input, max_shift_tensor);
        }
        offset++;

        // Test with empty tensors
        if ((Data[offset % Size] % 6) == 0) {
            torch::Tensor empty_input = torch::empty({0}, input_dtype);
            torch::Tensor empty_other = torch::empty({0}, input_dtype);
            torch::Tensor empty_result = torch::bitwise_right_shift(empty_input, empty_other);
        }
        offset++;

        // Test with single element tensors
        if ((Data[offset % Size] % 7) == 0) {
            torch::Tensor single_input = torch::randint(-128, 127, {1}, input_dtype);
            torch::Tensor single_other = torch::randint(0, 7, {1}, input_dtype);
            torch::Tensor single_result = torch::bitwise_right_shift(single_input, single_other);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}