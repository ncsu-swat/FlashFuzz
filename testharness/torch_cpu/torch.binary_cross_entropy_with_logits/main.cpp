#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (logits)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat && 
            input.scalar_type() != torch::kDouble) {
            input = input.to(torch::kFloat);
        }
        
        // Create target tensor with same shape as input
        // For BCE, target should be in [0, 1]
        torch::Tensor target = torch::rand_like(input);
        
        // If we have more data, use it to influence target values
        if (offset < Size) {
            torch::Tensor raw_target = fuzzer_utils::createTensor(Data, Size, offset);
            raw_target = raw_target.to(input.scalar_type());
            // Use sigmoid to map to [0, 1] range
            raw_target = torch::sigmoid(raw_target);
            
            // Try to use raw_target if shapes match or can be broadcast
            try {
                if (raw_target.numel() >= input.numel()) {
                    target = raw_target.flatten().slice(0, 0, input.numel()).reshape(input.sizes());
                } else {
                    // Use the generated random target
                }
            } catch (...) {
                // Keep the random target on any error
            }
        }
        
        // Determine reduction method
        int64_t reduction = 1; // Mean by default (0=None, 1=Mean, 2=Sum)
        if (offset < Size) {
            reduction = Data[offset++] % 3;
        }
        
        // Determine whether to use weight
        bool use_weight = false;
        torch::Tensor weight;
        if (offset < Size && Data[offset++] % 2 == 0) {
            use_weight = true;
            // Weight should have same shape as input
            weight = torch::rand_like(input) + 0.1f; // Positive weights
        }
        
        // Determine whether to use pos_weight
        bool use_pos_weight = false;
        torch::Tensor pos_weight;
        if (offset < Size && Data[offset++] % 2 == 0 && input.numel() > 0) {
            use_pos_weight = true;
            // pos_weight should be broadcastable - use scalar or match last dim
            if (offset < Size && Data[offset++] % 2 == 0) {
                // Use scalar pos_weight
                float pw_val = 0.1f + static_cast<float>(offset < Size ? Data[offset++] : 1) / 25.5f;
                pos_weight = torch::tensor({pw_val}, input.options());
            } else if (input.dim() > 0) {
                // Use 1D tensor matching last dimension
                int64_t last_dim = input.size(-1);
                pos_weight = torch::rand({last_dim}, input.options()) + 0.1f;
            } else {
                pos_weight = torch::tensor({1.0f}, input.options());
            }
        }
        
        // Call binary_cross_entropy_with_logits with functional API
        torch::Tensor output;
        
        // Helper lambda to call BCE with logits with proper reduction
        auto call_bce_with_logits = [&](bool with_weight, bool with_pos_weight) {
            auto options = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions();
            
            if (with_weight) {
                options = options.weight(weight);
            }
            if (with_pos_weight) {
                options = options.pos_weight(pos_weight);
            }
            
            // Set reduction - use separate if statements to avoid ternary type issues
            if (reduction == 0) {
                options = options.reduction(torch::kNone);
            } else if (reduction == 1) {
                options = options.reduction(torch::kMean);
            } else {
                options = options.reduction(torch::kSum);
            }
            
            return torch::nn::functional::binary_cross_entropy_with_logits(
                input, target, options);
        };
        
        try {
            output = call_bce_with_logits(use_weight, use_pos_weight);
        } catch (...) {
            // Shape mismatch or other expected errors - silently ignore
            return 0;
        }
        
        // Force computation
        if (output.defined() && output.numel() > 0) {
            volatile float val = output.sum().item<float>();
            (void)val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}