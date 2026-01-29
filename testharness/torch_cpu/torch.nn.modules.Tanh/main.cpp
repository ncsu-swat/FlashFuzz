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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Tanh module
        torch::nn::Tanh tanh_module;
        
        // Apply Tanh operation via module forward
        torch::Tensor output = tanh_module->forward(input);
        
        // Alternative way to apply tanh using functional interface
        torch::Tensor output2 = torch::tanh(input);
        
        // Try in-place version if tensor is floating point
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            input_copy.tanh_();
        }
        
        // Try with different options based on fuzzer data
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with contiguous tensor
            if (option_byte & 0x01) {
                torch::Tensor contiguous_input = input.contiguous();
                torch::Tensor contiguous_output = tanh_module->forward(contiguous_input);
            }
            
            // Try with non-contiguous tensor if possible
            if ((option_byte & 0x02) && input.dim() > 1 && input.size(0) > 1) {
                try {
                    torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                    torch::Tensor transposed_output = tanh_module->forward(transposed);
                } catch (...) {
                    // Silently ignore shape-related failures
                }
            }
            
            // Try with requires_grad set
            if ((option_byte & 0x04) && input.is_floating_point()) {
                torch::Tensor grad_input = input.clone().set_requires_grad(true);
                torch::Tensor grad_output = tanh_module->forward(grad_input);
                // Trigger backward pass
                try {
                    grad_output.sum().backward();
                } catch (...) {
                    // Silently ignore autograd failures
                }
            }
            
            // Try with different dtypes
            if (option_byte & 0x08) {
                try {
                    torch::Tensor float_input = input.to(torch::kFloat32);
                    torch::Tensor float_output = tanh_module->forward(float_input);
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            }
            
            // Try with double precision
            if (option_byte & 0x10) {
                try {
                    torch::Tensor double_input = input.to(torch::kFloat64);
                    torch::Tensor double_output = tanh_module->forward(double_input);
                } catch (...) {
                    // Silently ignore dtype conversion failures
                }
            }
            
            // Test with zero-dimensional tensor (scalar)
            if (option_byte & 0x20) {
                try {
                    torch::Tensor scalar = torch::tensor(0.5);
                    torch::Tensor scalar_output = tanh_module->forward(scalar);
                } catch (...) {
                    // Silently ignore failures
                }
            }
            
            // Test with empty tensor
            if (option_byte & 0x40) {
                try {
                    torch::Tensor empty = torch::empty({0});
                    torch::Tensor empty_output = tanh_module->forward(empty);
                } catch (...) {
                    // Silently ignore failures
                }
            }
        }
        
        // Additional coverage: test with special values
        if (offset + 1 < Size && input.is_floating_point()) {
            uint8_t special_byte = Data[offset++];
            
            if (special_byte & 0x01) {
                try {
                    // Test with infinity
                    torch::Tensor inf_tensor = torch::full_like(input, std::numeric_limits<float>::infinity());
                    torch::Tensor inf_output = tanh_module->forward(inf_tensor);
                } catch (...) {
                    // Silently ignore
                }
            }
            
            if (special_byte & 0x02) {
                try {
                    // Test with NaN
                    torch::Tensor nan_tensor = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                    torch::Tensor nan_output = tanh_module->forward(nan_tensor);
                } catch (...) {
                    // Silently ignore
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // Keep the input
}