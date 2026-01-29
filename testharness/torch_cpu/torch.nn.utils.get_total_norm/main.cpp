#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>

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
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 8) {
            return 0;
        }
        
        // Parse number of tensors to create (1-5)
        uint8_t num_tensors = (Data[offset++] % 5) + 1;
        
        // Parse norm type selector
        uint8_t norm_selector = Data[offset++];
        
        // Determine norm_type based on selector
        double norm_type;
        switch (norm_selector % 4) {
            case 0:
                norm_type = 1.0;  // L1 norm
                break;
            case 1:
                norm_type = 2.0;  // L2 norm (default)
                break;
            case 2:
                norm_type = std::numeric_limits<double>::infinity();  // Inf norm
                break;
            default:
                // Use a value from fuzzer data
                if (offset + sizeof(float) <= Size) {
                    float f_norm;
                    std::memcpy(&f_norm, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    // Clamp to reasonable range to avoid numerical issues
                    if (std::isfinite(f_norm) && f_norm > 0.0f && f_norm < 100.0f) {
                        norm_type = static_cast<double>(f_norm);
                    } else {
                        norm_type = 2.0;
                    }
                } else {
                    norm_type = 2.0;
                }
                break;
        }
        
        // Create a vector of tensors with gradients
        std::vector<torch::Tensor> parameters;
        
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                // Convert to float and enable gradients
                tensor = tensor.to(torch::kFloat32).clone().detach().requires_grad_(true);
                
                // Create a simple gradient by doing a backward pass
                // We need actual gradients for clip_grad_norm_ to work
                torch::Tensor loss = tensor.sum();
                loss.backward();
                
                parameters.push_back(tensor);
            } catch (...) {
                // Silently skip tensors that fail to create
                continue;
            }
        }
        
        // Skip if no valid tensors were created
        if (parameters.empty()) {
            return 0;
        }
        
        // Get max_norm from fuzzer data
        double max_norm = 1.0;
        if (offset + sizeof(float) <= Size) {
            float f_max;
            std::memcpy(&f_max, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(f_max) && f_max > 0.0f) {
                max_norm = static_cast<double>(std::abs(f_max));
            }
        }
        
        // Test clip_grad_norm_ which computes total norm internally
        // Using infinity as max_norm effectively just computes the norm without clipping
        // Note: clip_grad_norm_ returns a double, not a Tensor
        try {
            double total_norm = torch::nn::utils::clip_grad_norm_(
                parameters, 
                std::numeric_limits<double>::infinity(),
                norm_type
            );
            
            // Access the result to ensure computation is performed
            volatile double norm_value = total_norm;
            (void)norm_value;
        } catch (...) {
            // Silently handle expected failures (e.g., empty gradients)
        }
        
        // Re-create gradients for actual clipping test
        for (auto& param : parameters) {
            if (param.grad().defined()) {
                param.grad().zero_();
            }
            torch::Tensor loss = param.sum();
            loss.backward();
        }
        
        // Test with actual gradient clipping
        try {
            double clipped_norm = torch::nn::utils::clip_grad_norm_(
                parameters,
                max_norm,
                norm_type
            );
            
            volatile double clip_value = clipped_norm;
            (void)clip_value;
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Also test clip_grad_value_ for additional coverage
        if (offset < Size) {
            double clip_value = static_cast<double>(Data[offset++]) / 25.5;  // 0.0 to 10.0
            if (clip_value > 0.0) {
                // Re-create gradients
                for (auto& param : parameters) {
                    if (param.grad().defined()) {
                        param.grad().zero_();
                    }
                    torch::Tensor loss = param.sum();
                    loss.backward();
                }
                
                try {
                    torch::nn::utils::clip_grad_value_(parameters, clip_value);
                } catch (...) {
                    // Silently handle expected failures
                }
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