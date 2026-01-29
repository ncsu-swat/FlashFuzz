#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Create input tensor (predictions) - needs to be float for loss computation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        if (input.numel() == 0) {
            return 0;
        }
        input = input.to(torch::kFloat32).requires_grad_(true);
        
        // Create target tensor with same shape as input
        // SoftMarginLoss expects target values to be -1 or 1
        torch::Tensor target;
        if (offset < Size) {
            torch::Tensor raw_target = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape to match input shape and convert to -1 or 1 values
            target = torch::sign(raw_target.to(torch::kFloat32));
            // Replace zeros with 1
            target = torch::where(target == 0, torch::ones_like(target), target);
        } else {
            // If we don't have enough data, create a target with the same shape as input
            target = torch::ones_like(input);
        }
        
        // Ensure target has same shape as input
        try {
            target = target.expand_as(input).contiguous();
        } catch (...) {
            target = torch::ones_like(input);
        }
        
        // Create reduction mode from remaining data
        torch::nn::SoftMarginLossOptions::reduction_t reduction = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            if (reduction_selector == 0) {
                reduction = torch::kNone;
            } else if (reduction_selector == 1) {
                reduction = torch::kMean;
            } else {
                reduction = torch::kSum;
            }
        }
        
        // Create SoftMarginLoss module with the selected reduction mode
        torch::nn::SoftMarginLossOptions options;
        options.reduction(reduction);
        torch::nn::SoftMarginLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn(input, target);
        
        // Try backward pass if possible
        if (output.numel() > 0 && output.requires_grad()) {
            try {
                if (output.dim() == 0) {
                    output.backward();
                } else {
                    output.sum().backward();
                }
            } catch (...) {
                // Backward may fail for some configurations, ignore
            }
        }
        
        // Try with default options
        torch::nn::SoftMarginLoss default_loss_fn;
        torch::Tensor default_output = default_loss_fn(input.detach().requires_grad_(false), target);
        
        // Try direct functional version
        torch::Tensor functional_output = torch::nn::functional::soft_margin_loss(
            input.detach(), target, 
            torch::nn::functional::SoftMarginLossFuncOptions().reduction(reduction));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}