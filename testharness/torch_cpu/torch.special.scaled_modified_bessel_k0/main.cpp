#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the scaled_modified_bessel_k0 operation
        torch::Tensor result = torch::special::scaled_modified_bessel_k0(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            result.item();
        }
        
        // Try with different input types
        if (offset + 2 < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            torch::Tensor result2 = torch::special::scaled_modified_bessel_k0(input2);
            
            if (result2.defined() && result2.numel() > 0) {
                result2.item();
            }
        }
        
        // Try with edge cases if we have enough data
        if (Size > offset + 4) {
            // Create a tensor with potentially extreme values
            torch::Tensor extreme_values;
            
            // Use the remaining data to create a tensor that might have extreme values
            try {
                extreme_values = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                
                // Multiply by a large factor to get extreme values
                extreme_values = extreme_values * 1e10;
                
                torch::Tensor extreme_result = torch::special::scaled_modified_bessel_k0(extreme_values);
                
                if (extreme_result.defined() && extreme_result.numel() > 0) {
                    extreme_result.item();
                }
            } catch (const std::exception &) {
                // Continue if this specific edge case fails
            }
            
            // Try with negative values
            try {
                torch::Tensor negative_values = -torch::abs(extreme_values);
                torch::Tensor negative_result = torch::special::scaled_modified_bessel_k0(negative_values);
                
                if (negative_result.defined() && negative_result.numel() > 0) {
                    negative_result.item();
                }
            } catch (const std::exception &) {
                // Continue if this specific edge case fails
            }
            
            // Try with NaN and Inf values
            try {
                torch::Tensor special_values = torch::full_like(extreme_values, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor special_result = torch::special::scaled_modified_bessel_k0(special_values);
                
                if (special_result.defined() && special_result.numel() > 0) {
                    special_result.item();
                }
            } catch (const std::exception &) {
                // Continue if this specific edge case fails
            }
            
            try {
                torch::Tensor inf_values = torch::full_like(extreme_values, std::numeric_limits<float>::infinity());
                torch::Tensor inf_result = torch::special::scaled_modified_bessel_k0(inf_values);
                
                if (inf_result.defined() && inf_result.numel() > 0) {
                    inf_result.item();
                }
            } catch (const std::exception &) {
                // Continue if this specific edge case fails
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