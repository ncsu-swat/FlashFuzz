#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor parameters
        auto tensor_params = generate_tensor_params(Data, Size, offset);
        if (!tensor_params.has_value()) {
            return 0;
        }

        auto [shape, dtype, device] = tensor_params.value();
        
        // Create input tensor with various data patterns
        torch::Tensor input;
        
        // Try different tensor creation strategies based on remaining data
        if (offset < Size) {
            uint8_t strategy = Data[offset++];
            
            switch (strategy % 6) {
                case 0: {
                    // Normal random values
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                }
                case 1: {
                    // Values around zero (critical for erfc)
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 0.1;
                    break;
                }
                case 2: {
                    // Large positive values
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)).abs() + 5.0;
                    break;
                }
                case 3: {
                    // Large negative values
                    input = -(torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)).abs() + 5.0);
                    break;
                }
                case 4: {
                    // Special values including potential infinities
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 100.0;
                    break;
                }
                case 5: {
                    // Values from fuzzer data
                    input = create_tensor_from_data(Data, Size, offset, shape, dtype, device);
                    break;
                }
            }
        } else {
            input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Test torch::erfc with the input tensor
        torch::Tensor result = torch::erfc(input);
        
        // Verify result properties
        if (result.sizes() != input.sizes()) {
            std::cerr << "Size mismatch in erfc result" << std::endl;
        }
        
        // Test with different input modifications
        if (offset < Size) {
            uint8_t test_variant = Data[offset++];
            
            switch (test_variant % 4) {
                case 0: {
                    // Test with requires_grad
                    if (input.dtype().is_floating_point()) {
                        input.requires_grad_(true);
                        torch::Tensor grad_result = torch::erfc(input);
                        
                        // Test backward pass
                        if (grad_result.numel() > 0) {
                            torch::Tensor grad_output = torch::ones_like(grad_result);
                            grad_result.backward(grad_output);
                        }
                    }
                    break;
                }
                case 1: {
                    // Test with cloned input
                    torch::Tensor cloned_input = input.clone();
                    torch::Tensor cloned_result = torch::erfc(cloned_input);
                    break;
                }
                case 2: {
                    // Test with detached input
                    torch::Tensor detached_input = input.detach();
                    torch::Tensor detached_result = torch::erfc(detached_input);
                    break;
                }
                case 3: {
                    // Test with contiguous input
                    torch::Tensor contiguous_input = input.contiguous();
                    torch::Tensor contiguous_result = torch::erfc(contiguous_input);
                    break;
                }
            }
        }

        // Test edge cases with special values if floating point
        if (input.dtype().is_floating_point() && offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 3) {
                case 0: {
                    // Test with zeros
                    torch::Tensor zero_input = torch::zeros_like(input);
                    torch::Tensor zero_result = torch::erfc(zero_input);
                    break;
                }
                case 1: {
                    // Test with ones
                    torch::Tensor ones_input = torch::ones_like(input);
                    torch::Tensor ones_result = torch::erfc(ones_input);
                    break;
                }
                case 2: {
                    // Test with negative ones
                    torch::Tensor neg_ones_input = -torch::ones_like(input);
                    torch::Tensor neg_ones_result = torch::erfc(neg_ones_input);
                    break;
                }
            }
        }

        // Test in-place operation if supported
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            torch::Tensor inplace_input = input.clone();
            inplace_input.erfc_();
        }

        // Force evaluation of lazy tensors
        result.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}