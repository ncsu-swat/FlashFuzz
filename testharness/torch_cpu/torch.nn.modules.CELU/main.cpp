#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse alpha parameter from the remaining data
        float alpha = 1.0f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure alpha is not NaN or infinity
            if (std::isnan(alpha) || std::isinf(alpha)) {
                alpha = 1.0f;
            }
            
            // Ensure alpha is positive (CELU requires positive alpha)
            if (alpha <= 0) {
                alpha = std::abs(alpha);
                if (alpha == 0) alpha = 1.0f;
            }
            
            // Clamp alpha to reasonable range to avoid numerical issues
            if (alpha > 100.0f) alpha = 100.0f;
            if (alpha < 0.001f) alpha = 0.001f;
        }
        
        // Parse inplace flag
        bool inplace = false;
        if (offset < Size) {
            inplace = (Data[offset++] & 0x01) == 1;
        }
        
        // For inplace operations, we need a float tensor that is contiguous
        torch::Tensor input_for_module = input.to(torch::kFloat).contiguous();
        if (inplace) {
            // Clone to avoid modifying original for functional test later
            input_for_module = input_for_module.clone();
        }
        
        // Create CELU module with the parsed alpha
        torch::nn::CELU celu_module(torch::nn::CELUOptions().alpha(alpha).inplace(inplace));
        
        // Apply CELU to the input tensor
        torch::Tensor output = celu_module->forward(input_for_module);
        
        // Verify the output is valid
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Alternative approach: use the functional interface (non-inplace)
        torch::Tensor input_for_functional = input.to(torch::kFloat).contiguous();
        torch::Tensor output2 = torch::celu(input_for_functional, alpha);
        
        // Verify the output is valid
        if (output2.numel() > 0) {
            volatile float sum2 = output2.sum().item<float>();
            (void)sum2;
        }
        
        // Also test with negative input values to exercise both branches of CELU
        // CELU(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))
        torch::Tensor negative_input = -torch::abs(input.to(torch::kFloat).contiguous()) - 1.0f;
        torch::Tensor output3 = torch::celu(negative_input, alpha);
        
        if (output3.numel() > 0) {
            volatile float sum3 = output3.sum().item<float>();
            (void)sum3;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}