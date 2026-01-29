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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create a vector of parameters (tensors with gradients)
        std::vector<torch::Tensor> parameters;
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = (Data[offset++] % 4) + 1;
        
        // Create tensors with gradients
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            // Create a tensor with requires_grad=true
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure tensor is floating point for gradients
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32);
            }
            
            tensor = tensor.detach().requires_grad_(true);
            
            // Create a gradient tensor with the same shape
            torch::Tensor grad = torch::randn_like(tensor);
            
            // Try to use fuzzer data for gradient if available
            if (offset + 4 < Size) {
                try {
                    torch::Tensor fuzz_grad = fuzzer_utils::createTensor(Data, Size, offset);
                    if (!fuzz_grad.is_floating_point()) {
                        fuzz_grad = fuzz_grad.to(torch::kFloat32);
                    }
                    // Only use if element count matches
                    if (fuzz_grad.numel() == tensor.numel()) {
                        grad = fuzz_grad.reshape(tensor.sizes());
                    } else if (fuzz_grad.numel() > 0) {
                        // Scale to match
                        grad = fuzz_grad.flatten().slice(0, 0, std::min(fuzz_grad.numel(), tensor.numel()));
                        if (grad.numel() < tensor.numel()) {
                            grad = torch::cat({grad, torch::zeros(tensor.numel() - grad.numel())});
                        }
                        grad = grad.reshape(tensor.sizes());
                    }
                } catch (...) {
                    // Silently use randn_like gradient
                }
            }
            
            // Set the gradient
            tensor.mutable_grad() = grad;
            parameters.push_back(tensor);
        }
        
        // Skip if no parameters with gradients
        if (parameters.empty()) {
            return 0;
        }
        
        // Parse max_norm parameter
        double max_norm = 1.0;
        if (offset + sizeof(float) <= Size) {
            float raw_norm;
            std::memcpy(&raw_norm, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize: avoid NaN/Inf, use reasonable range
            if (!std::isnan(raw_norm) && !std::isinf(raw_norm)) {
                max_norm = static_cast<double>(raw_norm);
                // Clamp to reasonable range
                max_norm = std::max(-1e6, std::min(1e6, max_norm));
            }
        }
        
        // Parse norm_type parameter (common values: 0, 1, 2, inf)
        double norm_type = 2.0;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 5) {
                case 0: norm_type = 0.0; break;
                case 1: norm_type = 1.0; break;
                case 2: norm_type = 2.0; break;
                case 3: norm_type = std::numeric_limits<double>::infinity(); break;
                case 4: 
                    // Use a custom value from fuzzer data
                    if (offset + sizeof(float) <= Size) {
                        float raw_type;
                        std::memcpy(&raw_type, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        if (!std::isnan(raw_type) && !std::isinf(raw_type)) {
                            norm_type = static_cast<double>(raw_type);
                        }
                    }
                    break;
            }
        }
        
        // Parse error_if_nonfinite flag
        bool error_if_nonfinite = false;
        if (offset < Size) {
            error_if_nonfinite = Data[offset++] & 0x1;
        }
        
        // Call clip_grad_norm_ with vector of tensors
        try {
            double total_norm = torch::nn::utils::clip_grad_norm_(
                parameters, 
                max_norm,
                norm_type,
                error_if_nonfinite
            );
            
            // Use the result to prevent optimization
            if (std::isfinite(total_norm)) {
                volatile double sink = total_norm;
                (void)sink;
            }
        } catch (const c10::Error& e) {
            // Expected for some parameter combinations (e.g., error_if_nonfinite with inf grads)
        }
        
        // Test with different max_norm values if we have more data
        if (offset < Size && !parameters.empty()) {
            // Reset gradients for another test
            for (auto& p : parameters) {
                if (p.grad().defined()) {
                    p.mutable_grad() = torch::randn_like(p);
                }
            }
            
            // Try with absolute value of max_norm (common use case)
            try {
                double total_norm = torch::nn::utils::clip_grad_norm_(
                    parameters, 
                    std::abs(max_norm) + 0.1,  // Ensure positive
                    2.0,  // L2 norm
                    false
                );
                volatile double sink = total_norm;
                (void)sink;
            } catch (const c10::Error& e) {
                // Expected for some cases
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