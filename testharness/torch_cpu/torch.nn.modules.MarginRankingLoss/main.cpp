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
        // Need at least some data to create tensors and parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensors for MarginRankingLoss
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input1 and input2 have the same shape for proper comparison
        try {
            input2 = input2.reshape(input1.sizes());
        } catch (...) {
            // If reshape fails, create input2 with same shape
            input2 = torch::randn(input1.sizes());
        }
        
        // Create target tensor with values -1 or 1
        // MarginRankingLoss expects target to be -1 or 1
        torch::Tensor target = torch::ones(input1.sizes());
        if (offset < Size) {
            // Use fuzzer data to determine target values
            for (int64_t i = 0; i < target.numel() && (offset + i) < Size; i++) {
                if (Data[offset + (i % (Size - offset))] % 2 == 0) {
                    target.flatten()[i] = -1.0f;
                }
            }
            offset += std::min(static_cast<size_t>(target.numel()), Size - offset);
        }
        target = target.reshape(input1.sizes());
        
        // Ensure tensors are float type for loss computation
        input1 = input1.to(torch::kFloat32);
        input2 = input2.to(torch::kFloat32);
        target = target.to(torch::kFloat32);
        
        // Get margin value from the remaining data
        float margin = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp margin to reasonable range to avoid NaN
            if (std::isnan(margin) || std::isinf(margin)) {
                margin = 0.0f;
            }
            margin = std::fmax(-10.0f, std::fmin(10.0f, margin));
        }
        
        // Get reduction mode from the remaining data
        uint8_t reduction_byte = 0;
        if (offset < Size) {
            reduction_byte = Data[offset++];
        }
        
        // Create MarginRankingLoss module with the parsed options
        torch::nn::MarginRankingLossOptions options;
        options.margin(margin);
        
        // Set reduction mode
        switch (reduction_byte % 3) {
            case 0:
                options.reduction(torch::kNone);
                break;
            case 1:
                options.reduction(torch::kMean);
                break;
            case 2:
                options.reduction(torch::kSum);
                break;
        }
        
        torch::nn::MarginRankingLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input1, input2, target);
        
        // Enable gradients and perform backward pass
        input1 = input1.clone().detach().requires_grad_(true);
        input2 = input2.clone().detach().requires_grad_(true);
        
        // Recompute loss with gradient-enabled inputs
        torch::Tensor loss_grad = loss_fn(input1, input2, target);
        
        // Perform backward pass
        if (loss_grad.numel() == 1) {
            loss_grad.backward();
        } else {
            // For reduction=None, need to sum before backward
            loss_grad.sum().backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}