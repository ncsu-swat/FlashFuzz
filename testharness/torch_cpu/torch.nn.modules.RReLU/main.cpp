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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for RReLU from the remaining data
        float lower = 0.125f;
        float upper = 0.3333f;
        bool inplace = false;
        
        // If we have more data, use it to set parameters
        if (offset + 8 <= Size) {
            // Extract lower bound (between 0 and 1)
            float raw_lower;
            std::memcpy(&raw_lower, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Handle NaN/Inf
            if (!std::isfinite(raw_lower)) {
                raw_lower = 0.0f;
            }
            lower = std::abs(raw_lower) / (std::abs(raw_lower) + 1.0f); // Normalize to [0,1]
            
            // Extract upper bound (between lower and 1)
            float raw_upper;
            std::memcpy(&raw_upper, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Handle NaN/Inf
            if (!std::isfinite(raw_upper)) {
                raw_upper = 0.0f;
            }
            upper = lower + (1.0f - lower) * (std::abs(raw_upper) / (std::abs(raw_upper) + 1.0f));
            
            // Extract inplace flag
            if (offset < Size) {
                inplace = Data[offset++] & 0x01;
            }
        }
        
        // Ensure lower <= upper
        if (lower > upper) {
            std::swap(lower, upper);
        }
        
        // Create RReLU module
        torch::nn::RReLU rrelu(
            torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(inplace)
        );
        
        // Set to training mode to test the randomized behavior
        rrelu->train();
        torch::Tensor output_train;
        try {
            output_train = rrelu->forward(input.clone());
        } catch (...) {
            // Some tensor types may not be supported
        }
        
        // Test in eval mode too
        rrelu->eval();
        torch::Tensor output_eval;
        try {
            output_eval = rrelu->forward(input.clone());
        } catch (...) {
            // Some tensor types may not be supported
        }
        
        // Test the functional version as well (torch::nn::functional::rrelu)
        try {
            auto func_options = torch::nn::functional::RReLUFuncOptions()
                .lower(lower)
                .upper(upper)
                .inplace(false)
                .training(true);
            torch::Tensor output_functional = torch::nn::functional::rrelu(input.clone(), func_options);
        } catch (...) {
            // Functional version might fail for certain inputs
        }
        
        // Test backward pass if possible (only for floating point types)
        if (input.is_floating_point() && input.numel() > 0) {
            try {
                // Create a fresh tensor with requires_grad
                torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
                
                // Create a non-inplace module for gradient computation
                torch::nn::RReLU rrelu_grad(
                    torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(false)
                );
                rrelu_grad->train();
                
                auto output = rrelu_grad->forward(grad_input);
                
                // Sum to get scalar for backward
                if (output.numel() > 0) {
                    auto sum = output.sum();
                    sum.backward();
                }
            } catch (...) {
                // Backward may fail for some configurations
            }
        }
        
        // Test with different inplace setting
        try {
            torch::nn::RReLU rrelu_no_inplace(
                torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(false)
            );
            rrelu_no_inplace->eval();
            auto out = rrelu_no_inplace->forward(input.clone());
        } catch (...) {
            // May fail for some inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}