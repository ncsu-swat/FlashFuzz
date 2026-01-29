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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Mish requires floating point tensors
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create Mish module
        torch::nn::Mish mish_module;
        
        // Apply Mish operation via module
        torch::Tensor output = mish_module->forward(input);
        
        // Try functional version using torch::mish
        torch::Tensor output_functional = torch::mish(input);
        
        // Try inplace version using torch::mish_
        if (offset < Size) {
            bool use_inplace = Data[offset++] & 0x1;
            if (use_inplace) {
                torch::Tensor input_copy = input.clone();
                torch::mish_(input_copy);
            }
        }
        
        // Try with different tensor dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::ScalarType dtype;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            if (input.numel() > 0 && input.defined()) {
                try {
                    torch::Tensor converted_input = input.to(dtype);
                    torch::Tensor converted_output = mish_module->forward(converted_input);
                } catch (const std::exception&) {
                    // Some dtype conversions might not be supported, that's fine
                }
            }
        }
        
        // Try with different shaped tensors
        if (offset + 4 < Size) {
            int dim1 = (Data[offset++] % 8) + 1;
            int dim2 = (Data[offset++] % 8) + 1;
            int dim3 = (Data[offset++] % 4) + 1;
            uint8_t shape_type = Data[offset++] % 4;
            
            torch::Tensor shaped_input;
            switch (shape_type) {
                case 0:
                    // 1D tensor
                    shaped_input = torch::randn({dim1 * dim2});
                    break;
                case 1:
                    // 2D tensor
                    shaped_input = torch::randn({dim1, dim2});
                    break;
                case 2:
                    // 3D tensor (batch, channels, length)
                    shaped_input = torch::randn({dim3, dim1, dim2});
                    break;
                case 3:
                    // 4D tensor (batch, channels, height, width)
                    shaped_input = torch::randn({dim3, dim1, dim2, dim2});
                    break;
            }
            
            torch::Tensor shaped_output = mish_module->forward(shaped_input);
            
            // Also test inplace on shaped tensor
            torch::mish_(shaped_input);
        }
        
        // Test with edge case: empty tensor
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                torch::Tensor empty_input = torch::empty({0}, torch::kFloat32);
                torch::Tensor empty_output = mish_module->forward(empty_input);
            } catch (const std::exception&) {
                // Empty tensors might not be supported
            }
        }
        
        // Test with scalar tensor
        if (offset < Size && (Data[offset++] & 0x1)) {
            torch::Tensor scalar_input = torch::tensor(static_cast<float>(Data[offset % Size]) / 255.0f);
            torch::Tensor scalar_output = mish_module->forward(scalar_input);
            
            // Also test inplace on scalar
            torch::Tensor scalar_copy = scalar_input.clone();
            torch::mish_(scalar_copy);
        }
        
        // Test requires_grad path
        if (offset < Size && (Data[offset++] & 0x1)) {
            torch::Tensor grad_input = input.clone().set_requires_grad(true);
            torch::Tensor grad_output = mish_module->forward(grad_input);
            
            // Compute backward pass
            if (grad_output.numel() > 0) {
                try {
                    grad_output.sum().backward();
                } catch (const std::exception&) {
                    // Backward might fail for some configurations
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}