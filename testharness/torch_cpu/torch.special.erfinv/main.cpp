#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input values are in valid range for erfinv: (-1, 1)
        // erfinv is only defined in the range (-1, 1)
        input = torch::clamp(input, -0.9999, 0.9999);
        
        // Apply torch.special.erfinv operation
        torch::Tensor output = torch::special::erfinv(input);
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                // Test with specific edge cases based on the byte value
                if (edge_case % 5 == 0) {
                    // Test with values very close to -1
                    torch::Tensor near_minus_one = torch::full_like(input, -0.9999999);
                    torch::Tensor result_near_minus_one = torch::special::erfinv(near_minus_one);
                }
                else if (edge_case % 5 == 1) {
                    // Test with values very close to 1
                    torch::Tensor near_one = torch::full_like(input, 0.9999999);
                    torch::Tensor result_near_one = torch::special::erfinv(near_one);
                }
                else if (edge_case % 5 == 2) {
                    // Test with zeros
                    torch::Tensor zeros = torch::zeros_like(input);
                    torch::Tensor result_zeros = torch::special::erfinv(zeros);
                }
                else if (edge_case % 5 == 3) {
                    // Test with NaN values if supported by the tensor type
                    if (input.is_floating_point()) {
                        torch::Tensor nan_tensor = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                        torch::Tensor result_nan = torch::special::erfinv(nan_tensor);
                    }
                }
                else {
                    // Test with a mix of values
                    torch::Tensor mixed = torch::linspace(-0.99, 0.99, input.numel(), input.options());
                    mixed = mixed.reshape_as(input);
                    torch::Tensor result_mixed = torch::special::erfinv(mixed);
                }
            }
            catch (...) {
                // Silently catch expected failures from edge cases
            }
        }
        
        // Try calling with different tensor options if we have more data
        if (offset + 1 < Size) {
            uint8_t option_selector = Data[offset++];
            
            try {
                // Create a new tensor with different options
                torch::Tensor alt_input;
                
                if (option_selector % 4 == 0 && input.numel() > 0) {
                    // Test with non-contiguous tensor
                    if (input.dim() > 1) {
                        alt_input = input.transpose(0, input.dim() - 1);
                        torch::Tensor result_noncontiguous = torch::special::erfinv(alt_input);
                    }
                }
                else if (option_selector % 4 == 1) {
                    // Test with different dtype if original is floating point
                    if (input.is_floating_point()) {
                        auto new_dtype = (input.dtype() == torch::kFloat) ? torch::kDouble : torch::kFloat;
                        alt_input = input.to(new_dtype);
                        torch::Tensor result_diff_dtype = torch::special::erfinv(alt_input);
                    }
                }
                else if (option_selector % 4 == 2) {
                    // Test with broadcasting
                    if (input.dim() > 0 && input.size(0) > 0) {
                        std::vector<int64_t> new_shape(input.dim(), 1);
                        new_shape[0] = input.size(0);
                        alt_input = input.reshape(new_shape);
                        torch::Tensor expanded = alt_input.expand_as(input);
                        torch::Tensor result_broadcast = torch::special::erfinv(expanded);
                    }
                }
                else {
                    // Test with requires_grad if floating point
                    if (input.is_floating_point()) {
                        alt_input = input.detach().clone().set_requires_grad(true);
                        torch::Tensor result_requires_grad = torch::special::erfinv(alt_input);
                        
                        // Test backward pass
                        if (result_requires_grad.numel() > 0) {
                            torch::Tensor grad_output = torch::ones_like(result_requires_grad);
                            result_requires_grad.backward(grad_output);
                        }
                    }
                }
            }
            catch (...) {
                // Silently catch expected failures from tensor option variations
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