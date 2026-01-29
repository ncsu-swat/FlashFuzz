#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // SiLU requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create SiLU module and apply it
        torch::nn::SiLU silu_module;
        torch::Tensor output = silu_module(input);
        
        // Alternative way to apply SiLU using functional API
        torch::Tensor output_functional = torch::nn::functional::silu(input);
        
        // Try with different input shapes and types
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to floating point if needed
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat32);
            }
            
            // Apply SiLU to the second tensor
            torch::Tensor output2 = silu_module(input2);
            
            // Try with inplace operation using silu_
            try {
                torch::Tensor input2_clone = input2.clone();
                input2_clone.silu_();  // In-place SiLU operation
            } catch (...) {
                // Inplace may fail for certain tensor configurations, ignore silently
            }
        }
        
        // Test with different floating point types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset % Size];
            torch::Tensor typed_input;
            
            try {
                if (dtype_selector % 3 == 0) {
                    typed_input = input.to(torch::kFloat64);
                } else if (dtype_selector % 3 == 1) {
                    typed_input = input.to(torch::kFloat16);
                } else {
                    typed_input = input.to(torch::kFloat32);
                }
                
                torch::Tensor typed_output = silu_module(typed_input);
            } catch (...) {
                // Some dtype conversions or operations may fail, ignore silently
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input.dim() >= 2) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor output_transposed = silu_module(transposed);
            } catch (...) {
                // May fail for certain shapes, ignore silently
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}