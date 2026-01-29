#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <vector>
#include <cmath>

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
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create a vector of parameters (tensors) with gradients
        std::vector<torch::Tensor> parameters;
        
        // Determine number of parameters to create (1-4)
        uint8_t num_params = (Data[offset] % 4) + 1;
        offset++;
        
        // Create parameters with gradients
        for (uint8_t i = 0; i < num_params && offset < Size; i++) {
            // Create a tensor with random data
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Skip if tensor is empty or has no elements
            if (tensor.numel() == 0) {
                continue;
            }
            
            // Make it require gradients - must be floating point
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat32);
            }
            tensor = tensor.detach().requires_grad_(true);
            
            // Create a gradient for the tensor with the same shape
            torch::Tensor grad = torch::randn_like(tensor);
            
            // Optionally introduce some variation in gradient values
            if (offset < Size) {
                float scale = static_cast<float>(Data[offset]) / 25.5f; // 0-10 range
                grad = grad * scale;
                offset++;
            }
            
            // Set the gradient
            tensor.mutable_grad() = grad;
            
            parameters.push_back(tensor);
        }
        
        // Skip if no parameters with gradients
        if (parameters.empty()) {
            return 0;
        }
        
        // Extract max_norm from the input data (use a sensible range)
        double max_norm = 1.0;
        if (offset < Size) {
            max_norm = static_cast<double>(Data[offset]) / 25.5; // 0-10 range
            offset++;
        }
        // Ensure max_norm is valid (positive, finite)
        if (!std::isfinite(max_norm) || max_norm <= 0) {
            max_norm = 1.0;
        }
        
        // Extract norm_type from the input data
        double norm_type = 2.0;
        if (offset < Size) {
            // Common norm types: 1, 2, inf, -inf, 0.5, etc.
            uint8_t norm_selector = Data[offset] % 6;
            offset++;
            switch (norm_selector) {
                case 0: norm_type = 1.0; break;
                case 1: norm_type = 2.0; break;
                case 2: norm_type = std::numeric_limits<double>::infinity(); break;
                case 3: norm_type = -std::numeric_limits<double>::infinity(); break;
                case 4: norm_type = 0.5; break;
                case 5: norm_type = 3.0; break;
            }
        }
        
        // Extract error_if_nonfinite flag
        bool error_if_nonfinite = false;
        if (offset < Size) {
            error_if_nonfinite = Data[offset] % 2 == 1;
            offset++;
        }
        
        // Call clip_grad_norm_ with vector of tensors
        try {
            torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite);
        } catch (const std::exception &) {
            // Expected for certain norm_type/error_if_nonfinite combinations
        }
        
        // Try with a single tensor
        if (!parameters.empty() && parameters[0].grad().defined()) {
            // Reset gradient for another test
            parameters[0].mutable_grad() = torch::randn_like(parameters[0]);
            try {
                torch::nn::utils::clip_grad_norm_(parameters[0], max_norm, norm_type, error_if_nonfinite);
            } catch (const std::exception &) {
                // Expected for certain combinations
            }
        }
        
        // Try with different max_norm values
        if (offset < Size) {
            double alt_max_norm = static_cast<double>(Data[offset]) / 10.0;
            if (alt_max_norm > 0 && std::isfinite(alt_max_norm)) {
                // Reset gradients
                for (auto& p : parameters) {
                    if (p.grad().defined()) {
                        p.mutable_grad() = torch::randn_like(p);
                    }
                }
                try {
                    torch::nn::utils::clip_grad_norm_(parameters, alt_max_norm, 2.0, false);
                } catch (const std::exception &) {
                    // Silently catch expected failures
                }
            }
            offset++;
        }
        
        // Test with error_if_nonfinite = true and intentionally non-finite gradients
        if (offset < Size && Data[offset] % 4 == 0) {
            std::vector<torch::Tensor> test_params;
            torch::Tensor t = torch::ones({3, 3}, torch::kFloat32).requires_grad_(true);
            torch::Tensor inf_grad = torch::full({3, 3}, std::numeric_limits<float>::infinity());
            t.mutable_grad() = inf_grad;
            test_params.push_back(t);
            
            try {
                // This should throw with error_if_nonfinite=true
                torch::nn::utils::clip_grad_norm_(test_params, 1.0, 2.0, true);
            } catch (const std::exception &) {
                // Expected behavior - non-finite gradients with error_if_nonfinite=true
            }
            offset++;
        }
        
        // Test with NaN gradients
        if (offset < Size && Data[offset] % 4 == 1) {
            std::vector<torch::Tensor> test_params;
            torch::Tensor t = torch::ones({2, 2}, torch::kFloat32).requires_grad_(true);
            torch::Tensor nan_grad = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN());
            t.mutable_grad() = nan_grad;
            test_params.push_back(t);
            
            try {
                torch::nn::utils::clip_grad_norm_(test_params, 1.0, 2.0, false);
            } catch (const std::exception &) {
                // May throw for certain conditions
            }
            offset++;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}