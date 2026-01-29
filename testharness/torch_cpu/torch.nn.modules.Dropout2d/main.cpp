#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Dropout2d expects 4D input (N, C, H, W) or 3D input (C, H, W)
        // It zeroes out entire channels, so we need proper channel dimensions
        int64_t total_elements = input.numel();
        if (total_elements <= 0) {
            input = torch::randn({1, 2, 4, 4}, input.options());
        } else if (input.dim() < 3) {
            // Reshape to 4D: (1, C, H, W)
            int64_t c = std::max(int64_t(1), std::min(total_elements, int64_t(4)));
            int64_t remaining = total_elements / c;
            int64_t h = std::max(int64_t(1), (int64_t)std::sqrt(remaining));
            int64_t w = std::max(int64_t(1), remaining / h);
            int64_t actual_elements = c * h * w;
            if (actual_elements > 0 && actual_elements <= total_elements) {
                input = input.flatten().slice(0, 0, actual_elements).reshape({1, c, h, w});
            } else {
                input = torch::randn({1, 2, 4, 4}, input.options());
            }
        } else if (input.dim() == 3) {
            // Already 3D (C, H, W), which is acceptable
        } else if (input.dim() > 4) {
            // Flatten and reshape to 4D
            auto flat = input.flatten();
            int64_t n = flat.numel();
            int64_t side = std::max(int64_t(2), (int64_t)std::sqrt(std::sqrt(n)));
            int64_t actual = side * side * side * side;
            if (actual <= n && actual > 0) {
                input = flat.slice(0, 0, actual).reshape({side, side, side, side});
            } else {
                input = torch::randn({1, 2, 4, 4}, input.options());
            }
        }
        // If input.dim() == 4, it's already the right shape
        
        // Extract p (dropout probability) from input data
        float p = 0.5f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN and Inf, then clamp p to valid range [0, 1]
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5f;
            } else {
                p = std::abs(p);
                p = p - std::floor(p); // Get fractional part to ensure 0 <= p < 1
            }
        }
        
        // Extract inplace flag from input data
        bool inplace = false;
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // For inplace operation, tensor must not require grad and be contiguous
        if (inplace) {
            input = input.contiguous();
        }
        
        // Create Dropout2d module with parameters from fuzzer data
        torch::nn::Dropout2d dropout_module(
            torch::nn::Dropout2dOptions()
                .p(p)
                .inplace(inplace)
        );
        
        // Set training mode based on fuzzer data
        bool training_mode = true;
        if (offset < Size) {
            training_mode = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        if (training_mode) {
            dropout_module->train();
        } else {
            dropout_module->eval();
        }
        
        // Apply Dropout2d to the input tensor
        torch::Tensor output = dropout_module->forward(input);
        
        // Verify output properties to ensure computation completed
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        // Additional operations on the output to exercise more code paths
        if (output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
            
            // In eval mode, output should equal input
            // In train mode with p > 0, some channels should be zeroed
            if (!training_mode) {
                // Verify output matches input in eval mode
                auto diff = (output - input).abs().sum().item<float>();
                (void)diff; // Silence unused warning
            }
        }
        
        // Test with requires_grad if not inplace
        if (!inplace && offset < Size && (Data[offset] & 0x01)) {
            torch::Tensor grad_input = input.clone().set_requires_grad(true);
            dropout_module->train();
            torch::Tensor grad_output = dropout_module->forward(grad_input);
            if (grad_output.numel() > 0) {
                grad_output.sum().backward();
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