#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a list of tensors to represent parameters with gradients
        std::vector<torch::Tensor> parameters;
        
        // Determine number of tensors to create (1-4 based on available data)
        uint8_t num_tensors = 1;
        if (offset < Size) {
            num_tensors = (Data[offset++] % 4) + 1;
        }
        
        // Create tensors with gradients
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            // Create a tensor with float type for gradients
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to float if needed (gradients require floating point)
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32);
            }
            
            // Make it require gradients
            tensor = tensor.detach().clone().requires_grad_(true);
            
            // Create a gradient for the tensor with the same shape
            torch::Tensor grad = torch::randn_like(tensor);
            
            // If we have more data, scale the gradient
            if (offset < Size) {
                float scale = static_cast<float>(Data[offset++]) / 25.5f;
                grad = grad * scale;
            }
            
            // Set the gradient
            tensor.mutable_grad() = grad;
            
            // Add to parameters list
            parameters.push_back(tensor);
        }
        
        // Skip if no parameters with gradients
        if (parameters.empty()) {
            return 0;
        }
        
        // Parse max_norm parameter (use bounded values to avoid issues)
        double max_norm = 1.0;
        if (offset + sizeof(float) <= Size) {
            float raw_val;
            std::memcpy(&raw_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize: use absolute value and clamp to reasonable range
            if (!std::isnan(raw_val) && !std::isinf(raw_val)) {
                max_norm = std::abs(raw_val);
                if (max_norm < 1e-6) max_norm = 1e-6;
                if (max_norm > 1e6) max_norm = 1e6;
            }
        }
        
        // Parse norm_type parameter (common values: 1.0, 2.0, inf)
        double norm_type = 2.0;
        if (offset < Size) {
            uint8_t norm_choice = Data[offset++] % 5;
            switch (norm_choice) {
                case 0: norm_type = 1.0; break;
                case 1: norm_type = 2.0; break;
                case 2: norm_type = std::numeric_limits<double>::infinity(); break;
                case 3: norm_type = 0.5; break;
                case 4: norm_type = 3.0; break;
            }
        }
        
        // Test clip_grad_norm_ - returns double (the total norm)
        try {
            double total_norm = torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type);
            (void)total_norm; // Use the result
        } catch (const std::exception &) {
            // Silently catch expected failures (e.g., invalid norm_type combinations)
        }
        
        // Reset gradients and test clip_grad_value_ as well
        for (auto& param : parameters) {
            if (param.grad().defined()) {
                param.mutable_grad() = torch::randn_like(param);
            }
        }
        
        if (offset < Size) {
            double clip_value = 1.0;
            if (offset + sizeof(float) <= Size) {
                float raw_val;
                std::memcpy(&raw_val, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (!std::isnan(raw_val) && !std::isinf(raw_val)) {
                    clip_value = std::abs(raw_val);
                    if (clip_value < 1e-6) clip_value = 1e-6;
                    if (clip_value > 1e6) clip_value = 1e6;
                }
            }
            
            try {
                torch::nn::utils::clip_grad_value_(parameters, clip_value);
            } catch (const std::exception &) {
                // Silently catch expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}