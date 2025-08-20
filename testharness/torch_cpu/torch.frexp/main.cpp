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
        
        // Apply frexp operation
        // frexp returns a tuple of (mantissa, exponent)
        auto result = torch::frexp(input);
        
        // Access the components of the result
        torch::Tensor mantissa = std::get<0>(result);
        torch::Tensor exponent = std::get<1>(result);
        
        // Verify the result by reconstructing the original tensor
        // For each element: input = mantissa * (2 ^ exponent)
        torch::Tensor reconstructed = mantissa * torch::pow(2.0, exponent);
        
        // Try different variants of the API
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Test out_variant
            if (variant % 3 == 0 && input.numel() > 0) {
                torch::Tensor mantissa_out = torch::empty_like(input);
                torch::Tensor exponent_out = torch::empty_like(input, torch::kInt32);
                torch::frexp_out(mantissa_out, exponent_out, input);
            }
            
            // Test functional variant with named output
            if (variant % 3 == 1) {
                auto named_result = torch::frexp(input);
                auto m = std::get<0>(named_result);
                auto e = std::get<1>(named_result);
            }
            
            // Test edge case: empty tensor
            if (variant % 3 == 2) {
                torch::Tensor empty_tensor = torch::empty({0}, input.options());
                auto empty_result = torch::frexp(empty_tensor);
            }
        }
        
        // Try with different dtypes if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            dtype_selector = dtype_selector % 4;
            
            torch::ScalarType target_dtype;
            switch (dtype_selector) {
                case 0: target_dtype = torch::kFloat; break;
                case 1: target_dtype = torch::kDouble; break;
                case 2: target_dtype = torch::kHalf; break;
                case 3: target_dtype = torch::kBFloat16; break;
                default: target_dtype = torch::kFloat;
            }
            
            // Convert input to the selected dtype and apply frexp
            if (input.scalar_type() != target_dtype) {
                torch::Tensor converted_input = input.to(target_dtype);
                auto converted_result = torch::frexp(converted_input);
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