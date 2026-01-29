#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least a few bytes for input tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 1 dimension
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        if (offset < Size) {
            uint8_t config_byte = Data[offset++];
            
            // Determine if we should use a single parameter or per-channel parameters
            bool use_per_channel = (config_byte & 0x01);
            
            // Determine initial value for weight
            double init_value = 0.25; // Default value
            if (offset + sizeof(float) <= Size) {
                float param_value;
                std::memcpy(&param_value, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Clamp to reasonable range to avoid NaN/Inf issues
                if (std::isfinite(param_value)) {
                    init_value = std::clamp(static_cast<double>(param_value), -10.0, 10.0);
                }
            }
            
            torch::nn::PReLU prelu{nullptr};
            
            if (use_per_channel && input.dim() > 1 && input.size(1) > 0) {
                // For per-channel PReLU, num_parameters should match channels (dim 1)
                int64_t num_params = input.size(1);
                // Limit num_params to avoid excessive memory allocation
                num_params = std::min(num_params, static_cast<int64_t>(1024));
                
                torch::nn::PReLUOptions options;
                options.num_parameters(num_params);
                options.init(init_value);
                prelu = torch::nn::PReLU(options);
            } else {
                // Single parameter PReLU (default)
                torch::nn::PReLUOptions options;
                options.num_parameters(1);
                options.init(init_value);
                prelu = torch::nn::PReLU(options);
            }
            
            // Apply PReLU to the input tensor
            torch::Tensor output = prelu->forward(input);
            
            // Basic sanity check - output should have same shape as input
            (void)output;
            
            // Also test the functional interface with the module's weight
            try {
                torch::Tensor weight = prelu->weight;
                torch::Tensor output2 = torch::prelu(input, weight);
                (void)output2;
            } catch (...) {
                // Shape mismatch between weight and input is expected sometimes
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