#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Create input tensor - use float type for rrelu
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point (rrelu requires float)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract parameters for rrelu from the remaining data
        double lower = 0.125;
        double upper = 0.3333333333333333;
        bool training = false;
        
        // If we have more data, use it to set parameters
        if (offset + sizeof(double) <= Size) {
            double raw_lower;
            std::memcpy(&raw_lower, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Handle NaN/Inf and normalize to valid range [0, 0.5]
            if (std::isnan(raw_lower) || std::isinf(raw_lower)) {
                raw_lower = 0.0;
            }
            lower = std::abs(raw_lower) / (std::abs(raw_lower) + 1.0) * 0.5;
        }
        
        if (offset + sizeof(double) <= Size) {
            double raw_upper;
            std::memcpy(&raw_upper, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Handle NaN/Inf and normalize to [lower, 1]
            if (std::isnan(raw_upper) || std::isinf(raw_upper)) {
                raw_upper = 0.0;
            }
            upper = lower + (std::abs(raw_upper) / (std::abs(raw_upper) + 1.0)) * (1.0 - lower);
        }
        
        // Ensure lower <= upper
        if (lower > upper) {
            std::swap(lower, upper);
        }
        
        if (offset < Size) {
            training = (Data[offset++] % 2 == 0);
        }
        
        // Apply rrelu operation using functional interface
        torch::Tensor output;
        
        if (offset < Size) {
            uint8_t api_variant = Data[offset++] % 3;
            
            switch (api_variant) {
                case 0:
                    // Functional interface with custom parameters
                    output = torch::nn::functional::rrelu(
                        input, 
                        torch::nn::functional::RReLUFuncOptions()
                            .lower(lower)
                            .upper(upper)
                            .training(training)
                    );
                    break;
                case 1:
                    // Functional interface with default parameters
                    output = torch::nn::functional::rrelu(input);
                    break;
                case 2:
                    // Functional interface in training mode
                    output = torch::nn::functional::rrelu(
                        input,
                        torch::nn::functional::RReLUFuncOptions()
                            .lower(lower)
                            .upper(upper)
                            .training(true)
                    );
                    break;
            }
        } else {
            // Default call
            output = torch::nn::functional::rrelu(
                input,
                torch::nn::functional::RReLUFuncOptions()
                    .lower(lower)
                    .upper(upper)
                    .training(training)
            );
        }
        
        // Also test the inplace version via functional
        try {
            torch::Tensor input_copy = input.clone();
            output = torch::nn::functional::rrelu(
                input_copy,
                torch::nn::functional::RReLUFuncOptions()
                    .lower(lower)
                    .upper(upper)
                    .training(training)
                    .inplace(true)
            );
        } catch (...) {
            // Silently ignore inplace failures (e.g., if input requires grad)
        }
        
        // Force computation by accessing output properties
        volatile int64_t numel = output.numel();
        volatile int64_t ndim = output.dim();
        (void)numel;
        (void)ndim;
        
        // Sum to ensure computation happens
        if (output.numel() > 0) {
            volatile float sum_val = output.sum().item<float>();
            (void)sum_val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}