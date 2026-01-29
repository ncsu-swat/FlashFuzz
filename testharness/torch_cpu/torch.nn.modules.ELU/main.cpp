#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Parse alpha parameter if we have more data
        double alpha = 1.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize alpha to avoid NaN/Inf issues
            if (std::isnan(alpha) || std::isinf(alpha)) {
                alpha = 1.0;
            }
            // Clamp to reasonable range
            alpha = std::max(-100.0, std::min(100.0, alpha));
        }
        
        // Parse inplace parameter if we have more data
        bool inplace = false;
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create ELU module with the parsed parameters
        torch::nn::ELU elu_module(torch::nn::ELUOptions().alpha(alpha).inplace(inplace));
        
        // For inplace operation, we need to clone the input
        torch::Tensor working_input = inplace ? input.clone() : input;
        
        // Apply the ELU operation
        torch::Tensor output = elu_module->forward(working_input);
        
        // Access output to ensure computation happens
        volatile float sum = output.sum().item<float>();
        (void)sum;
        
        // Try a different alpha value if we have more data
        if (offset + sizeof(double) <= Size) {
            double new_alpha;
            std::memcpy(&new_alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Sanitize new_alpha
            if (std::isnan(new_alpha) || std::isinf(new_alpha)) {
                new_alpha = 1.0;
            }
            new_alpha = std::max(-100.0, std::min(100.0, new_alpha));
            
            // Create a new ELU module with the new alpha
            torch::nn::ELU elu_module2(torch::nn::ELUOptions().alpha(new_alpha).inplace(false));
            torch::Tensor output2 = elu_module2->forward(input);
            
            volatile float sum2 = output2.sum().item<float>();
            (void)sum2;
        }
        
        // Test with inplace=true if we originally had inplace=false
        if (!inplace) {
            torch::nn::ELU elu_module_inplace(torch::nn::ELUOptions().alpha(alpha).inplace(true));
            
            // Clone the input since we're using inplace operation
            torch::Tensor input_clone = input.clone();
            torch::Tensor output_inplace = elu_module_inplace->forward(input_clone);
            
            volatile float sum_inplace = output_inplace.sum().item<float>();
            (void)sum_inplace;
        }
        
        // Test functional interface as well for coverage
        try {
            torch::Tensor func_output = torch::elu(input, alpha);
            volatile float func_sum = func_output.sum().item<float>();
            (void)func_sum;
        } catch (...) {
            // Silently ignore functional API failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}