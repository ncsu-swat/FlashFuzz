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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - CELU requires floating point
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point for CELU operation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract alpha parameter from the remaining data
        double alpha = 1.0; // Default value
        if (offset + sizeof(float) <= Size) {
            float alpha_f;
            std::memcpy(&alpha_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure alpha is positive as required by CELU
            alpha = static_cast<double>(std::abs(alpha_f));
            
            // Avoid extremely large or small values that might cause numerical issues
            if (alpha > 1e6 || !std::isfinite(alpha)) {
                alpha = 1e6;
            }
            if (alpha < 1e-6) {
                alpha = 1e-6;
            }
        }
        
        // Extract inplace flag from fuzzer data
        bool inplace = false;
        if (offset < Size) {
            inplace = (Data[offset] % 2) == 0;
            offset++;
        }
        
        // Create CELU module with options
        torch::nn::CELU celu_module(torch::nn::CELUOptions().alpha(alpha).inplace(false));
        
        // Apply CELU operation via module
        torch::Tensor output = celu_module->forward(input);
        
        // Use the functional version with different options
        try {
            torch::Tensor output_functional = torch::nn::functional::celu(
                input, 
                torch::nn::functional::CELUFuncOptions().alpha(alpha).inplace(false)
            );
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Try inplace version on a copy
        if (inplace && input.is_contiguous()) {
            try {
                torch::Tensor input_copy = input.clone();
                // Use functional inplace version
                torch::nn::functional::celu(
                    input_copy, 
                    torch::nn::functional::CELUFuncOptions().alpha(alpha).inplace(true)
                );
            } catch (...) {
                // Silently catch expected failures for inplace operations
            }
        }
        
        // Also test with different tensor configurations
        if (offset < Size && (Data[offset] % 3) == 0) {
            try {
                // Test with non-contiguous tensor
                torch::Tensor strided = input.transpose(0, input.dim() > 1 ? 1 : 0);
                torch::Tensor out_strided = celu_module->forward(strided);
            } catch (...) {
                // Silently catch expected failures
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}