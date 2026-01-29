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
        
        // Need at least a few bytes to create tensors and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor with gradient tracking enabled
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        input = input.clone().detach().requires_grad_(true);
        
        // Create target tensor (should be same shape as input, with values -1 or 1)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to match target shape to input shape
            try {
                if (target.numel() != input.numel()) {
                    target = torch::randn_like(input);
                } else {
                    target = target.view_as(input);
                }
            } catch (...) {
                target = torch::randn_like(input);
            }
        } else {
            target = torch::randn_like(input);
        }
        
        // HingeEmbeddingLoss expects target to be -1 or 1
        target = torch::sign(target);
        // Replace zeros with ones (sign of zero is zero)
        target = torch::where(target == 0, torch::ones_like(target), target);
        
        // Parse margin parameter (default is 1.0)
        float margin = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp margin to reasonable range to avoid numerical issues
            if (std::isnan(margin) || std::isinf(margin)) {
                margin = 1.0f;
            }
            margin = std::max(-100.0f, std::min(100.0f, margin));
        }
        
        // Parse reduction parameter (0: none, 1: mean, 2: sum)
        uint8_t reduction_mode = 1; // default to mean
        if (offset < Size) {
            reduction_mode = Data[offset] % 3;
            offset++;
        }
        
        torch::nn::HingeEmbeddingLossOptions options;
        options = options.margin(margin);
        
        // Set reduction
        switch (reduction_mode) {
            case 0:
                options = options.reduction(torch::kNone);
                break;
            case 1:
                options = options.reduction(torch::kMean);
                break;
            case 2:
                options = options.reduction(torch::kSum);
                break;
        }
        
        // Create the loss function
        torch::nn::HingeEmbeddingLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input, target);
        
        // Force computation - handle both scalar and tensor outputs
        if (reduction_mode == 0) {
            // kNone returns a tensor, sum it to get a scalar for backward
            torch::Tensor loss_sum = loss.sum();
            if (loss_sum.numel() > 0 && !loss_sum.isnan().item<bool>() && 
                !loss_sum.isinf().item<bool>()) {
                loss_sum.backward();
            }
        } else {
            // kMean and kSum return scalars
            float loss_value = loss.item<float>();
            (void)loss_value; // Suppress unused warning
            
            if (!std::isnan(loss_value) && !std::isinf(loss_value)) {
                loss.backward();
            }
        }
        
        // Access gradient to ensure it was computed
        if (input.grad().defined()) {
            auto grad_sum = input.grad().sum().item<float>();
            (void)grad_sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}