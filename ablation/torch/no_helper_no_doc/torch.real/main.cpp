#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor configuration
        auto tensor_config = parseTensorConfig(Data, Size, offset);
        if (!tensor_config.has_value()) {
            return 0;
        }

        auto config = tensor_config.value();
        
        // Create input tensor - torch.real works on complex tensors
        torch::Tensor input;
        
        // Generate complex tensor for testing torch.real
        if (config.dtype == torch::kComplexFloat || config.dtype == torch::kComplexDouble) {
            input = generateTensor(config);
        } else {
            // Convert to complex type for testing
            auto real_tensor = generateTensor(config);
            if (config.dtype == torch::kFloat || config.dtype == torch::kDouble || 
                config.dtype == torch::kHalf || config.dtype == torch::kBFloat16) {
                // Create complex tensor from real tensor
                auto zero_imag = torch::zeros_like(real_tensor);
                input = torch::complex(real_tensor, zero_imag);
            } else {
                // For integer types, convert to float first then to complex
                auto float_tensor = real_tensor.to(torch::kFloat);
                auto zero_imag = torch::zeros_like(float_tensor);
                input = torch::complex(float_tensor, zero_imag);
            }
        }

        // Test torch.real with various scenarios
        
        // Basic torch.real operation
        auto result1 = torch::real(input);
        
        // Test with cloned tensor
        auto input_clone = input.clone();
        auto result2 = torch::real(input_clone);
        
        // Test with detached tensor
        auto input_detached = input.detach();
        auto result3 = torch::real(input_detached);
        
        // Test with contiguous tensor
        if (!input.is_contiguous()) {
            auto input_contiguous = input.contiguous();
            auto result4 = torch::real(input_contiguous);
        }
        
        // Test with transposed tensor (if 2D or higher)
        if (input.dim() >= 2) {
            auto input_transposed = input.transpose(0, 1);
            auto result5 = torch::real(input_transposed);
        }
        
        // Test with sliced tensor
        if (input.numel() > 1) {
            auto input_sliced = input.flatten().slice(0, 0, std::min(input.numel(), 10L));
            auto result6 = torch::real(input_sliced);
        }
        
        // Test with reshaped tensor
        if (input.numel() > 1) {
            auto input_reshaped = input.reshape({-1});
            auto result7 = torch::real(input_reshaped);
        }
        
        // Test with tensor that requires gradient (if applicable)
        if (input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble) {
            auto input_grad = input.clone().requires_grad_(true);
            auto result8 = torch::real(input_grad);
            
            // Test backward pass if result is scalar or small
            if (result8.numel() == 1) {
                result8.backward();
            } else if (result8.numel() <= 10) {
                auto grad_output = torch::ones_like(result8);
                result8.backward(grad_output);
            }
        }
        
        // Test with different memory layouts
        if (input.dim() >= 2) {
            // Test with channels_last format (if 4D)
            if (input.dim() == 4) {
                auto input_channels_last = input.to(torch::MemoryFormat::ChannelsLast);
                auto result9 = torch::real(input_channels_last);
            }
        }
        
        // Verify properties of results
        if (result1.defined()) {
            // Check that result is real-valued
            assert(!result1.is_complex());
            
            // Check shape preservation
            assert(result1.sizes() == input.sizes());
            
            // Check device consistency
            assert(result1.device() == input.device());
        }
        
        // Test edge cases with special values
        if (input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble) {
            // Create tensor with special complex values
            auto special_input = input.clone();
            if (special_input.numel() > 0) {
                // Set some elements to special values
                auto flat = special_input.flatten();
                if (flat.numel() >= 1) {
                    flat[0] = std::complex<float>(INFINITY, 0.0f);
                }
                if (flat.numel() >= 2) {
                    flat[1] = std::complex<float>(-INFINITY, 1.0f);
                }
                if (flat.numel() >= 3) {
                    flat[2] = std::complex<float>(NAN, 2.0f);
                }
                if (flat.numel() >= 4) {
                    flat[3] = std::complex<float>(0.0f, INFINITY);
                }
                
                auto special_result = torch::real(special_input);
            }
        }
        
        // Test with empty tensor
        auto empty_complex = torch::empty({0}, torch::dtype(torch::kComplexFloat));
        auto empty_result = torch::real(empty_complex);
        
        // Test with scalar tensor
        auto scalar_complex = torch::tensor(std::complex<float>(3.14f, 2.71f));
        auto scalar_result = torch::real(scalar_complex);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}