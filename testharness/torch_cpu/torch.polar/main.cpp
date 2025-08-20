#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two tensors for abs and angle inputs to polar
        torch::Tensor abs_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Try with a simple tensor instead
            abs_tensor = torch::abs(abs_tensor);
            torch::Tensor angle_tensor = torch::zeros_like(abs_tensor);
            torch::Tensor result = torch::polar(abs_tensor, angle_tensor);
            return 0;
        }
        
        torch::Tensor angle_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure abs is non-negative as required by polar
        // We don't want to filter out all negative cases, so we'll use abs
        // This allows testing with zeros and other edge cases
        abs_tensor = torch::abs(abs_tensor);
        
        // Try different variants of polar
        // 1. Basic polar call
        torch::Tensor result1 = torch::polar(abs_tensor, angle_tensor);
        
        // 2. Try with broadcasting if shapes don't match
        if (abs_tensor.sizes() != angle_tensor.sizes()) {
            // Create a scalar tensor for one of the inputs to test broadcasting
            if (Size > offset) {
                uint8_t selector = Data[offset++] % 3;
                if (selector == 0) {
                    // Use scalar for abs
                    double abs_val = abs_tensor.item<double>();
                    torch::Tensor scalar_abs = torch::tensor(abs_val);
                    torch::Tensor result2 = torch::polar(scalar_abs, angle_tensor);
                } else if (selector == 1) {
                    // Use scalar for angle
                    double angle_val = angle_tensor.item<double>();
                    torch::Tensor scalar_angle = torch::tensor(angle_val);
                    torch::Tensor result2 = torch::polar(abs_tensor, scalar_angle);
                } else {
                    // Try with reshaped tensors
                    torch::Tensor reshaped_abs = abs_tensor.reshape({-1});
                    torch::Tensor reshaped_angle = angle_tensor.reshape({-1});
                    if (reshaped_abs.size(0) > 0 && reshaped_angle.size(0) > 0) {
                        int64_t min_size = std::min(reshaped_abs.size(0), reshaped_angle.size(0));
                        torch::Tensor result2 = torch::polar(
                            reshaped_abs.slice(0, 0, min_size), 
                            reshaped_angle.slice(0, 0, min_size)
                        );
                    }
                }
            }
        }
        
        // 3. Try with out variant if we have enough data
        if (Size > offset) {
            // Create output tensor with same shape as expected result
            torch::Tensor out_tensor = torch::empty_like(result1);
            torch::polar_out(out_tensor, abs_tensor, angle_tensor);
        }
        
        // 4. Try with different dtypes if we have enough data
        if (Size > offset) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::ScalarType dtype;
            
            switch (dtype_selector) {
                case 0:
                    dtype = torch::kFloat;
                    break;
                case 1:
                    dtype = torch::kDouble;
                    break;
                default:
                    dtype = torch::kComplexFloat;
                    break;
            }
            
            // Convert tensors to the selected dtype
            torch::Tensor abs_converted = abs_tensor.to(dtype);
            torch::Tensor angle_converted = angle_tensor.to(dtype);
            
            // Call polar with the converted tensors
            torch::Tensor result3 = torch::polar(abs_converted, angle_converted);
        }
        
        // 5. Try with empty tensors
        if (Size > offset) {
            uint8_t empty_selector = Data[offset++] % 3;
            
            if (empty_selector == 0) {
                // Empty abs tensor
                torch::Tensor empty_abs = torch::empty({0}, abs_tensor.options());
                torch::Tensor result4 = torch::polar(empty_abs, angle_tensor);
            } else if (empty_selector == 1) {
                // Empty angle tensor
                torch::Tensor empty_angle = torch::empty({0}, angle_tensor.options());
                torch::Tensor result4 = torch::polar(abs_tensor, empty_angle);
            } else {
                // Both empty
                torch::Tensor empty_abs = torch::empty({0}, abs_tensor.options());
                torch::Tensor empty_angle = torch::empty({0}, angle_tensor.options());
                torch::Tensor result4 = torch::polar(empty_abs, empty_angle);
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