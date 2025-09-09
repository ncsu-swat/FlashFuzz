#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and dtypes
        auto input_tensor = generateTensor(Data, Size, offset);
        
        // Test basic frac operation
        auto result = torch::frac(input_tensor);
        
        // Test in-place frac operation
        auto input_copy = input_tensor.clone();
        input_copy.frac_();
        
        // Test with different tensor types and edge cases
        if (offset < Size) {
            // Test with different dtypes if we have more data
            auto dtype_choice = consumeIntegralInRange<int>(Data, Size, offset, 0, 3);
            torch::Tensor typed_tensor;
            
            switch (dtype_choice) {
                case 0:
                    typed_tensor = input_tensor.to(torch::kFloat32);
                    break;
                case 1:
                    typed_tensor = input_tensor.to(torch::kFloat64);
                    break;
                case 2:
                    typed_tensor = input_tensor.to(torch::kHalf);
                    break;
                default:
                    typed_tensor = input_tensor.to(torch::kBFloat16);
                    break;
            }
            
            auto typed_result = torch::frac(typed_tensor);
        }
        
        // Test with special values if we have floating point tensor
        if (input_tensor.is_floating_point() && offset < Size) {
            auto special_tensor = torch::empty_like(input_tensor);
            auto flat_view = special_tensor.flatten();
            
            if (flat_view.numel() > 0) {
                // Fill with special values based on remaining data
                for (int64_t i = 0; i < flat_view.numel() && offset < Size; ++i) {
                    auto special_choice = consumeIntegralInRange<int>(Data, Size, offset, 0, 5);
                    float special_val;
                    
                    switch (special_choice) {
                        case 0: special_val = std::numeric_limits<float>::infinity(); break;
                        case 1: special_val = -std::numeric_limits<float>::infinity(); break;
                        case 2: special_val = std::numeric_limits<float>::quiet_NaN(); break;
                        case 3: special_val = 0.0f; break;
                        case 4: special_val = -0.0f; break;
                        default: special_val = consumeFloatingPoint<float>(Data, Size, offset); break;
                    }
                    
                    flat_view[i] = special_val;
                }
                
                auto special_result = torch::frac(special_tensor);
            }
        }
        
        // Test with different tensor properties
        if (offset < Size) {
            // Test with requires_grad if applicable
            if (input_tensor.is_floating_point()) {
                auto grad_tensor = input_tensor.clone().requires_grad_(true);
                auto grad_result = torch::frac(grad_tensor);
                
                // Test backward pass
                if (grad_result.numel() > 0) {
                    auto grad_output = torch::ones_like(grad_result);
                    grad_result.backward(grad_output);
                }
            }
        }
        
        // Test with different memory layouts
        if (input_tensor.dim() >= 2 && offset < Size) {
            auto transposed = input_tensor.transpose(0, 1);
            auto transposed_result = torch::frac(transposed);
            
            // Test with contiguous tensor
            auto contiguous = transposed.contiguous();
            auto contiguous_result = torch::frac(contiguous);
        }
        
        // Test output tensor variant if we have more data
        if (offset < Size) {
            auto output_tensor = torch::empty_like(input_tensor);
            torch::frac_out(output_tensor, input_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}