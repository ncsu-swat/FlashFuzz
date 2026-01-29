#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isfinite, std::abs

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
        
        // Need at least a few bytes for the fuzzer to work
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor with gradients
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point for gradient computation
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Make the tensor require gradients
        tensor = tensor.detach().clone().requires_grad_(true);
        
        // Create a simple operation to generate gradients
        torch::Tensor output = tensor.sum();
        
        // Backpropagate to compute gradients
        output.backward();
        
        // Extract max_norm parameter from the input data
        double max_norm = 1.0;
        if (offset + sizeof(float) <= Size) {
            float raw_max_norm;
            std::memcpy(&raw_max_norm, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize: ensure it's a valid positive number
            if (std::isfinite(raw_max_norm) && raw_max_norm > 0) {
                max_norm = static_cast<double>(raw_max_norm);
            }
        }
        
        // Extract norm_type parameter from the input data (common values: 1.0, 2.0, inf)
        double norm_type = 2.0;
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t norm_selector = Data[offset++];
            // Map to common norm types
            switch (norm_selector % 4) {
                case 0: norm_type = 1.0; break;  // L1 norm
                case 1: norm_type = 2.0; break;  // L2 norm (default)
                case 2: norm_type = std::numeric_limits<double>::infinity(); break;  // Inf norm
                case 3: norm_type = 0.5; break;  // Fractional norm
            }
        }
        
        // Create a vector of parameters
        std::vector<torch::Tensor> parameters;
        parameters.push_back(tensor);
        
        // Optionally add more tensors with gradients
        if (offset + 4 <= Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (!tensor2.is_floating_point()) {
                tensor2 = tensor2.to(torch::kFloat32);
            }
            tensor2 = tensor2.detach().clone().requires_grad_(true);
            torch::Tensor output2 = tensor2.sum();
            output2.backward();
            parameters.push_back(tensor2);
        }
        
        // Apply clip_grad_norm_ - returns double (total norm), not Tensor
        try {
            double total_norm = torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type);
            // Use the result to prevent optimization
            (void)total_norm;
        } catch (const std::exception &) {
            // Expected failures for invalid configurations
        }
        
        // Try clip_grad_value_ as well
        if (offset + sizeof(float) <= Size) {
            float raw_clip_value;
            std::memcpy(&raw_clip_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize: ensure it's a valid positive number
            double clip_value = 1.0;
            if (std::isfinite(raw_clip_value) && raw_clip_value > 0) {
                clip_value = static_cast<double>(raw_clip_value);
            }
            
            // Need to regenerate gradients since clip_grad_norm_ modified them
            for (auto& param : parameters) {
                if (param.grad().defined()) {
                    param.grad().zero_();
                }
                torch::Tensor out = param.sum();
                out.backward();
            }
            
            try {
                torch::nn::utils::clip_grad_value_(parameters, clip_value);
            } catch (const std::exception &) {
                // Expected failures for invalid configurations
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